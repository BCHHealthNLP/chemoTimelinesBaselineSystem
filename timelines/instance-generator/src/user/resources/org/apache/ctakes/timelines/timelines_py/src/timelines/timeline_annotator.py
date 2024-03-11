import os
import torch
import sys
import xmltodict
import pandas as pd

from itertools import chain
from transformers import pipeline
from ctakes_pbj.component import cas_annotator
from ctakes_pbj.type_system import ctakes_types
from typing import List, Tuple, Dict, Optional, Generator, Union, Set
from collections import defaultdict
from cassis.typesystem import (
    FeatureStructure,
)

from cassis.cas import Cas


DTR_WINDOW_RADIUS = 10
MAX_TLINK_DISTANCE = 60
TLINK_PAD_LENGTH = 2
MODEL_MAX_LEN = 512
CHEMO_TUI = "T061"
DTR_OUTPUT_COLUMNS = [
    "DCT",
    "patient_id",
    "chemo_text",
    "chemo_annotation_id",
    "dtr",
    "normed_timex",
    "timex_annotation_id",
    "tlink",
    "note_name",
    "dtr_inst",
    "tlink_inst",
]

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

TLINK_HF_HUB = "HealthNLP/pubmedbert_tlink"

DTR_HF_HUB = "HealthNLP/pubmedbert_dtr"

CONMOD_HF_HUB = "HealthNLP/pubmedbert_conmod"


class TimelineAnnotator(cas_annotator.CasAnnotator):
    def __init__(self):
        self.use_dtr = False
        self.use_conmod = False
        self.output_dir = "."
        self.dtr_classifier = lambda _: []
        self.tlink_classifier = lambda _: []
        self.conmod_classifier = lambda _: []
        self.raw_events = []
        self.anafora_dir = "."

    def init_params(self, arg_parser):
        self.use_dtr = arg_parser.use_dtr
        self.use_conmod = arg_parser.use_conmod
        self.output_dir = arg_parser.output_dir
        self.anafora_dir = arg_parser.anafora_dir

    def initialize(self):
        if torch.cuda.is_available():
            main_device = 0
            print("GPU with CUDA is available, using GPU")
        else:
            main_device = -1
            print("GPU with CUDA is not available, defaulting to CPU")

        self.tlink_classifier = TimelineAnnotator._get_pipeline(
            TLINK_HF_HUB,
            main_device,
        )

        print("TLINK classifier loaded")
        if self.use_dtr:
            self.dtr_classifier = TimelineAnnotator._get_pipeline(
                DTR_HF_HUB,
                main_device,
            )

            print("DTR classifier loaded")

        if self.use_conmod:
            self.conmod_classifier = TimelineAnnotator._get_pipeline(
                CONMOD_HF_HUB,
                main_device,
            )

            print("Conmod classifier loaded")

    def declare_params(self, arg_parser):
        arg_parser.add_arg("--use_dtr", action="store_true")
        arg_parser.add_arg("--use_conmod", action="store_true")
        arg_parser.add_arg("--anafora_dir", type=str)

    def process(self, cas: Cas):
        proc_mentions = TimelineAnnotator._get_event_mentions(cas, self.anafora_dir)

        if len(proc_mentions) > 0:
            self._write_raw_timelines(cas, proc_mentions)
        else:
            # empty discovery writing logic so no patients are skipped for the eval script
            self._add_empty_discovery(cas)
            patient_id, note_name = TimelineAnnotator._pt_and_note(cas)
            print(
                f"No chemotherapy mentions ( using TUI: {CHEMO_TUI} ) found in patient {patient_id} note {note_name}  - skipping"
            )

    def collection_process_complete(self):
        output_columns = DTR_OUTPUT_COLUMNS if self.use_dtr else NO_DTR_OUTPUT_COLUMNS
        output_tsv_name = "unsummarized_output.tsv"
        output_path = "".join([self.output_dir, "/", output_tsv_name])
        print("Finished processing notes")
        print(f"Writing results for all input in {output_path}")
        pt_df = pd.DataFrame.from_records(
            self.raw_events,
            columns=output_columns,
        )
        pt_df.to_csv(output_path, index=False, sep="\t")
        print("Finished writing")
        sys.exit()

    def _write_raw_timelines(self, cas: Cas, proc_mentions: List[FeatureStructure]):
        patient_id, note_name = TimelineAnnotator._pt_and_note(cas)
        if not self.use_conmod:
            print(
                f"Modality filtering turned off, proceeding for patient {patient_id} note {note_name}"
            )
            self._write_actual_proc_mentions(cas, proc_mentions)
            return
        conmod_instances = (
            TimelineAnnotator._get_conmod_instance(chemo, cas)
            for chemo in proc_mentions
        )

        conmod_classifications = (
            result["label"]
            for result in filter(None, self.conmod_classifier(conmod_instances))
        )
        actual_proc_mentions = [
            chemo
            for chemo, modality in zip(proc_mentions, conmod_classifications)
            if modality == "ACTUAL"
        ]

        if len(actual_proc_mentions) > 0:
            print(
                f"Found concrete chemotherapy mentions in patient {patient_id} note {note_name} - proceeding"
            )
            self._write_actual_proc_mentions(cas, actual_proc_mentions)
        else:
            # empty discovery writing logic so no patients are skipped for the eval script
            self._add_empty_discovery(cas)
            print(
                f"No concrete chemotherapy mentions found in patient {patient_id} note {note_name} - skipping"
            )

    def _write_actual_proc_mentions(
        self, cas: Cas, positive_chemo_mentions: List[FeatureStructure]
    ):
        timex_type = cas.typesystem.get_type(ctakes_types.TimeMention)
        cas_source_data = cas.select(ctakes_types.Metadata)[0].sourceData
        document_creation_time = cas_source_data.sourceOriginalDate
        relevant_timexes = TimelineAnnotator._timexes_with_normalization(
            cas.select(timex_type)
        )

        base_tokens, token_map = TimelineAnnotator._tokens_and_map(cas, mode="dtr")
        begin2token, end2token = TimelineAnnotator._invert_map(token_map)

        def local_window_mentions(chemo):
            return TimelineAnnotator._get_tlink_window_mentions(
                chemo, relevant_timexes, begin2token, end2token, token_map
            )

        def dtr_result(chemo):
            inst = TimelineAnnotator._get_dtr_instance(
                chemo, base_tokens, begin2token, end2token
            )
            result = list(self.dtr_classifier(inst))[0]
            label = result["label"]
            return label, inst

        def tlink_result(chemo, timex):
            inst = TimelineAnnotator._get_tlink_instance(
                chemo, timex, base_tokens, begin2token, end2token
            )
            result = list(self.tlink_classifier(inst))[0]
            label = result["label"]
            if timex.begin < chemo.begin:
                label = LABEL_TO_INVERTED_LABEL[label]
            return label, inst

        def tlink_result_dict(chemo):
            window_mentions = local_window_mentions(chemo)
            return {
                window_mention: tlink_result(chemo, window_mention)
                for window_mention in window_mentions
            }

        patient_id, note_name = TimelineAnnotator._pt_and_note(cas)

        # Needed for Jiarui's deduplication algorithm
        annotation_ids = {
            annotation: f"{index}@e@{note_name}@system"
            for index, annotation in enumerate(
                sorted(
                    chain.from_iterable((positive_chemo_mentions, relevant_timexes)),
                    key=lambda annotation: annotation.begin,
                )
            )
        }
        if (
            len(list(relevant_timexes)) == 0
            or len(
                list(
                    chain.from_iterable(
                        map(local_window_mentions, positive_chemo_mentions)
                    )
                )
            )
            == 0
        ):
            print(
                f"WARNING: No timexes suitable for TLINK pairing discovered in {patient_id} file {note_name}"
            )
            self._add_empty_discovery(cas)
            return
        for chemo in positive_chemo_mentions:
            chemo_dtr, dtr_inst = "", ""  # purely so pyright stops complaining
            if self.use_dtr:
                chemo_dtr, dtr_inst = dtr_result(chemo)
            tlink_dict = tlink_result_dict(chemo)
            for timex, tlink_inst_pair in tlink_dict.items():
                tlink, tlink_inst = tlink_inst_pair
                chemo_text = TimelineAnnotator._normalize_mention(chemo)
                timex_text = timex.time.normalizedForm
                if self.use_dtr:
                    instance = [
                        document_creation_time,
                        patient_id,
                        chemo_text,
                        annotation_ids[chemo],
                        chemo_dtr,
                        timex_text,
                        annotation_ids[timex],
                        tlink,
                        note_name,
                        dtr_inst,
                        tlink_inst,
                    ]
                else:
                    instance = [
                        document_creation_time,
                        patient_id,
                        chemo_text,
                        annotation_ids[chemo],
                        timex_text,
                        annotation_ids[timex],
                        tlink,
                        note_name,
                        tlink_inst,
                    ]
                self.raw_events.append(instance)

    def _add_empty_discovery(self, cas: Cas):
        cas_source_data = cas.select(ctakes_types.Metadata)[0].sourceData
        document_creation_time = cas_source_data.sourceOriginalDate
        patient_id, note_name = TimelineAnnotator._pt_and_note(cas)
        self.raw_events.append(
            TimelineAnnotator._empty_discovery(
                document_creation_time, patient_id, note_name, self.use_dtr
            )
        )

    @staticmethod
    def _get_event_mentions(cas: Cas, anafora_dir: str) -> List[FeatureStructure]:
        event_type = cas.typesystem.get_type(ctakes_types.Event)
        _, note_name = TimelineAnnotator._pt_and_note(cas)
        # this is extension agnostic but inefficient
        def relevant_path(doc_path: str) -> bool:
            base_name = os.path.basename(doc_path).split(".")[0]
            return base_name.lower() == note_name.lower()

        def full_path(f: str) -> str:
            return os.path.join(anafora_dir, f)

        relevants = filter(relevant_path, map(full_path, os.listdir(anafora_dir)))
        relevant_file = next(relevants, None)
        if relevant_file is None:
            return []
        additional = next(relevants, None)
        if additional is not None:
            print(
                f"Error: multiple Anafora files found for patient note {note_name}, at least two {relevant_file} and {additional}"
            )
            return []
        return []

    @staticmethod
    def anafora_entities(xml_path: str) -> List[FeatureStructure]:
        with open(xml_path) as fr:
            xml_data_dict = xmltodict.parse(fr.read())

        if xml_data_dict["data"]["annotations"] is None:
            return []

        if (
            not(xml_data_dict["data"]["annotations"]
            and xml_data_dict["data"]["annotations"]["entity"])
        ):
            return []
        if isinstance(xml_data_dict["data"]["annotations"]["entity"], list):
            entities = [
                ent
                for ent in xml_data_dict["data"]["annotations"]["entity"]
                if ent["type"] != "Markable"
            ]
        else:
            entities = [xml_data_dict["data"]["annotations"]["entity"]]
            entities = [ent for ent in entities if ent["type"] != "Markable"]
        entity_with_duplicate = defaultdict(list)
        entity_no_duplicate = []
        for ent in entities:
            entity_with_duplicate[ent["span"]].append(ent)
        for ents in entity_with_duplicate.values():
            if len(ents) == 1:
                entity_no_duplicate.append(ents[0])
            else:
                valid_ent = get_ent_with_doctimerel(ents)
                entity_no_duplicate.append(valid_ent)
        for ent in entity_no_duplicate:
            ## Some entities' IDs don't match their text name
            # I'm assuming there are no issues like this in the packaged
            # dev or test, will ask Jiarui about it if I run into any counter-evidence
            # if (
            #     ent["id"] == "1@e@ID195_SUM_08-27-2010_1@gaby"
            #     and "ID177_CON_07-29-2010_2" in txt_path
            # ):
            #     ent["id"] = "1@e@ID177_CON_07-29-2010_2@gaby"
            # if (
            #     ent["id"] == "1@e@ID152_CON_02-24-2011_1@gaby"
            #     and "ID152_CON_02-14-2011_1" in txt_path
            # ):
            #     ent["id"] = "1@e@ID152_CON_02-14-2011_1@gaby"
            # elif (
            #     "1@e@ID168_SV_03-29-2012_1@gaby" == ent["id"]
            #     and "ID168_SV_03-29-2012_2" in txt_path
            # ):
            #     ent["id"] = "1@e@ID168_SV_03-29-2012_2@gaby"
            # elif (
            #     "3@e@ID168_SV_03-29-2012_1@gaby" == ent["id"]
            #     and "ID168_SV_03-29-2012_2" in txt_path
            # ):
            #     ent["id"] = "3@e@ID168_SV_03-29-2012_2@gaby"
            # elif (
            #     "1@e@ID169_SV_05-28-2015_1@gaby" == ent["id"]
            #     and "ID169_SV_11-19-2015_1" in txt_path
            # ):
            #     ent["id"] = "1@e@ID169_SV_11-19-2015_1@gaby"
            # E.g. 152@e@ID057_path_168@gold
            ent_note_id = ent["id"].split("@")[-2]
            if "-" in ent_note_id:
                pt_id, pt_type, second, third = ent_note_id.split("_")
                month, day, year = second.split("-")
                ent_note_id = (
                    pt_id + "_" + pt_type + "_" + "_".join([month, day, year]) + "_" + third
                )
            cur_ent_char_token_map = token_char_idx_map[ent_note_id]
            if ent["type"] == "Markable":
                continue
            # if (
            #     ent_note_id in DOC_WT_BLACKSLASH_R["dev/breast/"]
            #     or ent_note_id in DOC_WT_BLACKSLASH_R["train/breast/"]
            # ):
            #     ent_node = get_a_thyme2_entity_node(
            #         ent,
            #         text_data_str_map[ent_note_id],
            #         cur_ent_char_token_map,
            #         eol_map=end_of_line_count[ent_note_id],
            #     )
            else:
                ent_node = get_a_thyme2_entity_node(
                    ent, text_data_str_map[ent_note_id], cur_ent_char_token_map
                )
            assert ent_node.start_char_idx is not None
            all_entity_nodes.append(ent_node)
            return []

    @staticmethod
    def _empty_discovery(
        DCT: str, patient_id: str, note_name: str, use_dtr: bool
    ) -> List[str]:
        if use_dtr:
            return [
                DCT,
                patient_id,
                "none",
                "none",
                "none",
                "none",
                "none",
                "none",
                note_name,
                "none",
                "none",
            ]
        return [
            DCT,
            patient_id,
            "none",
            "none",
            "none",
            "none",
            "none",
            note_name,
            "none",
        ]

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
        # duplicates = defaultdict(list)
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

    # previous conmod model used Pitt sentencing and tokenization
    # for the next conmod model it should use DTR instances
    @staticmethod
    def _get_conmod_instance(event: FeatureStructure, cas: Cas) -> str:
        raw_sentence = list(cas.select_covering(ctakes_types.Sentence, event))[0]
        tokens, token_map = TimelineAnnotator._tokens_and_map(
            cas, raw_sentence, mode="conmod"
        )
        begin2token, end2token = TimelineAnnotator._invert_map(token_map)
        event_begin = begin2token[event.begin]
        event_end = end2token[event.end] + 1
        str_builder = (
            tokens[:event_begin]
            + ["<e>"]
            + tokens[event_begin:event_end]
            + ["</e>"]
            + tokens[event_end:]
        )
        result = " ".join(str_builder)
        return result

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
    def _get_dtr_instance(
        event: FeatureStructure,
        tokens: List[str],
        begin2token: Dict[int, int],
        end2token: Dict[int, int],
    ) -> str:
        event_begin = begin2token[event.begin]
        event_end = end2token[event.end] + 1
        str_builder = (
            tokens[event_begin - DTR_WINDOW_RADIUS : event_begin]
            + ["<e>"]
            + tokens[event_begin:event_end]
            + ["</e>"]
            + tokens[event_end : event_end + DTR_WINDOW_RADIUS]
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
    ) -> Generator[FeatureStructure, None, None]:
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

        for mention in relevant_mentions:
            if in_window(mention):
                yield mention

    @staticmethod
    def _deleted_neighborhood(
        central_mention: FeatureStructure, mentions: List[FeatureStructure]
    ) -> Generator[FeatureStructure, None, None]:
        for mention in mentions:
            if central_mention != mention:
                yield mention

    @staticmethod
    def _pt_and_note(cas: Cas):
        document_path_collection = cas.select(ctakes_types.DocumentPath)
        document_path = list(document_path_collection)[0].documentPath
        note_name = os.path.basename(document_path).split(".")[0]
        patient_id = note_name.split("_")[0]
        return patient_id, note_name

    @staticmethod
    def _get_tuis(event: FeatureStructure) -> Set[str]:
        def get_tui(event):
            return getattr(event, "tui", None)

        ont_concept_arr = getattr(event, "ontologyConceptArr", None)
        elements = getattr(ont_concept_arr, "elements", [])
        if len(elements) == 0:
            return set()
        return {tui for tui in map(get_tui, elements) if tui is not None}

    @staticmethod
    def _get_pipeline(path, device):
        return pipeline(
            model=path,
            device=device,
            padding=True,
            truncation=True,
            max_length=MODEL_MAX_LEN,
        )
