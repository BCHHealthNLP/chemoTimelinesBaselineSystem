import os
import re
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

import pandas as pd
from cassis.cas import Cas
from cassis.typesystem import FeatureStructure
from ctakes_pbj.component import cas_annotator
from ctakes_pbj.type_system import ctakes_types
from more_itertools import flatten, unzip

from .ModelInterface import ClassificationModelInterface, TaggingModelInterface

MAX_TLINK_DISTANCE = 60
TLINK_PAD_LENGTH = 2
MODEL_MAX_LEN = 512

NO_DTR_OUTPUT_COLUMNS = [
    "DCT",
    "patient_id",
    "chemo_text",
    "chemo_annotation_id",
    "normed_timex",
    "timex_annotation_id",
    "tlink",
    "note_name",
    "tlink_inst",
]
LABEL_TO_INVERTED_LABEL = {
    "before": "after",
    "after": "before",
    "begins-on": "ends-on",
    "ends-on": "begins-on",
    "overlap": "overlap",
    "contains": "contains-1",
    "noted-on": "noted-on-1",
    "contains-1": "contains",
    "noted-on-1": "noted-on",
    "contains-subevent": "contains-subevent-1",
    "contains-subevent-1": "contains-subevent",
    "none": "none",
}


class TimelineAnnotator(cas_annotator.CasAnnotator):
    def __init__(self):
        self.output_dir = "."
        self.raw_events = deque()

    def init_params(self, arg_parser):
        self.tlink_model_path = arg_parser.tlink_model_path
        self.med_model_path = arg_parser.med_model_path
        self.output_dir = arg_parser.output_dir

    def initialize(self):
        # For trainer objects we don't need this
        # since it binds to CUDA by default
        # if torch.cuda.is_available():
        #     main_device = 0
        #     print("GPU with CUDA is available, using GPU")
        # else:
        #     main_device = -1
        #     print("GPU with CUDA is not available, defaulting to CPU")

        self.tlink_classifier = ClassificationModelInterface(self.tlink_model_path)
        print("TLINK classifier loaded")
        self.med_tagger = TaggingModelInterface(self.med_model_path)
        print("Medication tagger loaded")

    def declare_params(self, arg_parser):
        arg_parser.add_arg("--tlink_model_path")
        arg_parser.add_arg("--med_model_path")

    def process(self, cas: Cas):
        timex_type = cas.typesystem.get_type(ctakes_types.TimeMention)
        event_type = cas.typesystem.get_type(ctakes_types.Event)
        event_mention_type = cas.typesystem.get_type(ctakes_types.EventMention)
        event_properties_type = cas.typesystem.get_type(ctakes_types.EventProperties)
        patient_id, note_name = TimelineAnnotator._pt_and_note(cas)
        # sorting here so we have a reliable way of accessing the
        # chemo - timex pairs later
        relevant_timexes = TimelineAnnotator._timexes_with_normalization(
            sorted(cas.select(timex_type), key=lambda t: t.begin)
        )
        if len(relevant_timexes) == 0:
            print(
                f"No normalized time expressions for {patient_id} {note_name} - skipping"
            )
            return
        chemo_mentions = self._get_chemo_mentions(cas)
        if len(chemo_mentions) == 0:
            print(
                f"No chemotherapy mentions detected for {patient_id} {note_name} - skipping"
            )
            return

        def insert_chemo_mention(chemo_mention: Tuple[int, int]) -> None:
            begin, end = chemo_mention
            event_mention = event_mention_type(begin=begin, end=end)
            cas.add(event_mention)
            event_properties = event_properties_type()
            event = event_type()
            setattr(event, "properties", event_properties)
            setattr(event_mention, "event", event)

        for chemo_mention in chemo_mentions:
            insert_chemo_mention(chemo_mention)

        base_tokens, token_map = TimelineAnnotator._tokens_and_map(cas, mode="dtr")
        begin2token, end2token = TimelineAnnotator._invert_map(token_map)

        def local_window_mentions(
            chemo: FeatureStructure,
        ) -> List[FeatureStructure]:
            return TimelineAnnotator._get_tlink_window_mentions(
                chemo, relevant_timexes, begin2token, end2token, token_map
            )

        chemo_to_relevant_timexes = {
            chemo: local_window_mentions(chemo)
            for chemo in sorted(cas.select(event_type), key=lambda t: t.begin)
        }

        if not any(chemo_to_relevant_timexes.values()):
            print(
                f"No compatible normalized timexes found for any of the chemos in {patient_id} {note_name} - skipping"
            )
            return
        ordered_chemo_timex_pairs = [
            (chemo, timex)
            for chemo, timexes in chemo_to_relevant_timexes.items()
            for timex in timexes
        ]
        tlink_classification_instances = (
            TimelineAnnotator._get_tlink_instance(
                chemo, timex, base_tokens, begin2token, end2token
            )
            for chemo, timex in ordered_chemo_timex_pairs
        )

        raw_tlink_classifications = self._get_tlink_classifications(
            tlink_classification_instances
        )

        cas_source_data = cas.select(ctakes_types.Metadata)[0].sourceData
        document_creation_time = cas_source_data.sourceOriginalDate
        for pair, tlink in zip(ordered_chemo_timex_pairs, raw_tlink_classifications):
            chemo, timex = pair
            if timex.begin < chemo.begin:
                tlink = LABEL_TO_INVERTED_LABEL[tlink]
            self.raw_events.append(
                (
                    patient_id,
                    note_name,
                    document_creation_time,
                    TimelineAnnotator._normalize_mention(chemo),
                    TimelineAnnotator._normalize_mention(timex),
                    tlink,
                )
            )

    def collection_process_complete(self):
        output_columns = [
            "patient_identifier",
            "note_identifier",
            "document_creation_time",
            "chemo_mention_text",
            "normalized_timex_text",
            "tlink",
        ]
        output_tsv_name = "unsummarized_output.tsv"
        output_path = "".join((self.output_dir, "/", output_tsv_name))
        print("Finished processing notes")
        print(f"Writing results for all input in {output_path}")
        pt_df = pd.DataFrame.from_records(
            self.raw_events,
            columns=output_columns,
        )
        pt_df.to_csv(output_path, index=False, sep="\t")
        print("Finished writing")

    @staticmethod
    def _ctakes_tokenize(
        cas: Cas, sentence: FeatureStructure
    ) -> List[FeatureStructure]:
        return sorted(
            cas.select_covered(ctakes_types.BaseToken, sentence), key=lambda t: t.begin
        )

    @staticmethod
    def _ctakes_clean(
        cas: Cas, sentence: FeatureStructure
    ) -> Tuple[str, List[Tuple[int, int]]]:
        base_tokens = deque()
        token_map = deque()
        newline_tokens = cas.select_covered(ctakes_types.NewlineToken, sentence)
        newline_token_indices = {(item.begin, item.end) for item in newline_tokens}

        for base_token in TimelineAnnotator._ctakes_tokenize(cas, sentence):
            if (
                (base_token.begin, base_token.end)
                not in newline_token_indices
                # and base_token.get_covered_text()
                # and not base_token.get_covered_text().isspace()
            ):
                base_tokens.append(base_token.get_covered_text())
                token_map.append((base_token.begin, base_token.end))
            else:
                # since these indices are tracked as well in the RT code
                base_tokens.append("<cr>")
                token_map.append((base_token.begin, base_token.end))
        return " ".join(base_tokens), list(token_map)

    @staticmethod
    def _normalize_mention(mention: Union[FeatureStructure, None]) -> str:
        if mention is None:
            return "ERROR"
        raw_mention_text = mention.get_covered_text()
        return raw_mention_text.replace("\n", "")

    @staticmethod
    def _tokens_and_map(
        cas: Cas, context: Optional[FeatureStructure] = None, mode="conmod"
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        base_tokens = []
        token_map = []
        newline_tag = "<cr>" if mode == "conmod" else "<newline>"
        newline_tokens = cas.select(ctakes_types.NewlineToken)
        newline_token_indices = {(item.begin, item.end) for item in newline_tokens}
        raw_token_collection = (
            cas.select(ctakes_types.BaseToken)
            if context is None
            else cas.select_covered(ctakes_types.BaseToken, context)
        )
        token_collection: Dict[int, Tuple[int, str]] = {}
        for base_token in raw_token_collection:
            begin = base_token.begin
            end = base_token.end
            token_text = (
                base_token.get_covered_text()
                if (begin, end) not in newline_token_indices
                else newline_tag
            )
            # TODO - ask Sean and Guergana why there might be duplicate Newline tokens
            # if begin in token_collection:
            #     prior_end, prior_text = token_collection[begin]
            #     print(
            #         f"WARNING: two tokens {(token_text, begin, end)} and {(prior_text, begin, prior_end)} share the same begin index, overwriting with latest"
            #     )
            token_collection[begin] = (end, token_text)
        for begin in sorted(token_collection):
            end, token_text = token_collection[begin]
            base_tokens.append(token_text)
            token_map.append((begin, end))

        return base_tokens, token_map

    @staticmethod
    def _tags_to_indices(tagged_sentence: str) -> List[Tuple[int, int]]:
        span_begin, span_end = 0, 0
        indices = []
        # Group B's individually as well as B's followed by
        # any nummber of I's, e.g.
        # OOOOOOBBBBBBBIIIIBIBIBI
        # -> OOOOOO B B B B B B BIIII BI BI BI
        for span in filter(None, re.split(r"(BI*)", tagged_sentence)):
            span_end = len(span) + span_begin - 1
            if span[0] == "B":
                # Get indices in list/string of each span
                # which describes a mention
                indices.append((span_begin, span_end))
            span_begin = span_end + 1
        return indices

    @staticmethod
    def _invert_map(
        token_map: List[Tuple[int, int]]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        begin_map: Dict[int, int] = {}
        end_map: Dict[int, int] = {}
        for token_index, token_boundaries in enumerate(token_map):
            begin, end = token_boundaries
            # these warnings are kind of by-passed by previous logic
            # since any time two tokens shared a begin and or an end
            # it was always a newline token and its exact duplicate
            if begin in begin_map:
                print(
                    f"pre-existing token begin entry {begin} -> {begin_map[begin]} against {token_index} in reverse token map"
                )
                print(
                    f"full currently stored token info {begin_map[begin]} -> {token_map[begin_map[begin]]} against current candidate {token_index} -> {(begin, end)}"
                )

            if end in end_map:
                print(
                    f"pre-existing token end entry {end} -> {end_map[end]} against {token_index} in reverse token map"
                )
                print(
                    f"full currently stored token info {end_map[end]} -> {token_map[end_map[end]]} against current candidate {token_index} -> {(begin, end)}"
                )
            begin_map[begin] = token_index
            end_map[end] = token_index
        return begin_map, end_map

    def _get_chemo_mentions(self, cas: Cas) -> List[Tuple[int, int]]:
        sentence_type = cas.typesystem.get_type(ctakes_types.Sentence)
        cas_sentences = sorted(cas.select(sentence_type), key=lambda t: t.begin)
        sentence_texts, index_maps = unzip(
            TimelineAnnotator._ctakes_clean(cas, sentence) for sentence in cas_sentences
        )
        sentence_tags = self.med_tagger.process_instances(
            cast(Iterable[str], sentence_texts)
        )
        # the whole point of using unzip(...) instead of zip(*...)
        # was for type support but you get what you pay for
        index_maps = cast(Iterable[List[Tuple[int, int]]], index_maps)

        def to_character_indices(
            tagged_sentence: str, index_map: List[Tuple[int, int]]
        ) -> Iterable[Tuple[int, int]]:
            tag_groups = TimelineAnnotator._tags_to_indices(tagged_sentence)
            for tag_group in tag_groups:
                token_begin, token_end = tag_group
                yield index_map[token_begin][0], index_map[token_end][1]

        return list(
            flatten(
                to_character_indices(tagged_sentence, index_map)
                for tagged_sentence, index_map in zip(sentence_tags, index_maps)
            )
        )

    def _get_tlink_classifications(
        self, tlink_classification_instances: Iterable[str]
    ) -> List[str]:
        return self.tlink_classifier(tlink_classification_instances)

    @staticmethod
    def _timexes_with_normalization(
        timexes: List[FeatureStructure],
    ) -> List[FeatureStructure]:
        def relevant(timex):
            return hasattr(timex, "time") and hasattr(timex.time, "normalizedForm")

        return [timex for timex in timexes if relevant(timex)]

    @staticmethod
    def _get_tlink_instance(
        event: FeatureStructure,
        timex: FeatureStructure,
        tokens: List[str],
        begin2token: Dict[int, int],
        end2token: Dict[int, int],
    ) -> str:
        # Have an event and a timex/other event which are up to 60 tokens apart from each other
        # have two tokens before first annotation, first annotation plus tags
        # then all the text between the two annotations
        # second annotation plus tags, the last two tokens after the second annotation
        event_begin = begin2token[event.begin]
        event_end = end2token[event.end] + 1
        event_tags = ("<e>", "</e>")
        event_packet = (event_begin, event_end, event_tags)
        timex_begin = begin2token[timex.begin]
        timex_end = end2token[timex.end] + 1
        timex_tags = ("<t>", "</t>")
        timex_packet = (timex_begin, timex_end, timex_tags)

        first_packet, second_packet = sorted(
            (event_packet, timex_packet), key=lambda s: s[0]
        )
        (first_begin, first_end, first_tags) = first_packet
        (
            first_open_tag,
            first_close_tag,
        ) = first_tags  # if is_timex else ("<e1>", "</e1>")

        (second_begin, second_end, second_tags) = second_packet
        (
            second_open_tag,
            second_close_tag,
        ) = second_tags  # if is_timex else ("<e2>", "</e2>")

        # to avoid wrap arounds
        start_token_idx = max(0, first_begin - TLINK_PAD_LENGTH)
        end_token_idx = min(len(tokens) - 1, second_end + TLINK_PAD_LENGTH)

        str_builder = (
            # first two tokens
            tokens[start_token_idx:first_begin]
            # tag body of the first mention
            + [first_open_tag]
            + tokens[first_begin:first_end]
            + [first_close_tag]
            # intermediate part of the window
            + tokens[first_end:second_begin]
            # tag body of the second mention
            + [second_open_tag]
            + tokens[second_begin:second_end]
            + [second_close_tag]
            # ending part of the window
            + tokens[second_end:end_token_idx]
        )
        result = " ".join(str_builder)
        return result

    @staticmethod
    def _get_tlink_window_mentions(
        event: FeatureStructure,
        relevant_mentions: List[FeatureStructure],
        begin2token: Dict[int, int],
        end2token: Dict[int, int],
        token2char: List[Tuple[int, int]],
    ) -> List[FeatureStructure]:
        event_begin_token_index = begin2token[event.begin]
        event_end_token_index = end2token[event.end]

        token_window_begin = max(0, event_begin_token_index - MAX_TLINK_DISTANCE)
        token_window_end = min(
            len(token2char) - 1, event_end_token_index + MAX_TLINK_DISTANCE
        )

        char_window_begin = token2char[token_window_begin][0]
        char_window_end = token2char[token_window_end][1]

        def in_window(mention):
            begin_inside = char_window_begin <= mention.begin <= char_window_end
            end_inside = char_window_begin <= mention.end <= char_window_end
            return begin_inside and end_inside

        return [mention for mention in relevant_mentions if in_window(mention)]

    @staticmethod
    def _pt_and_note(cas: Cas):
        document_path_collection = cas.select(ctakes_types.DocumentPath)
        document_path = list(document_path_collection)[0].documentPath
        note_name = os.path.basename(document_path).split(".")[0]
        patient_id = note_name.split("_")[0]
        return patient_id, note_name
