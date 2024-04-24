from abc import ABC, abstractmethod
from itertools import groupby
from typing import Iterable, List, Tuple, Union
from datasets import Dataset

import numpy as np
from more_itertools import unzip
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
)


class ModelInterface(ABC):
    def __init__(self, path: str) -> None:
        self.load_from_path(path)

    @abstractmethod
    def load_from_path(self, path: str) -> None:
        pass

    def __call__(self, instances: Union[str, Iterable[str]]):
        self.process_instances(instances)

    @abstractmethod
    def process_instances(self, instances: Iterable[str]) -> List[str]:
        pass


class ClassificationModelInterface(ModelInterface):
    def load_from_path(self, path: str) -> None:
        self.config = AutoConfig.from_pretrained(
            path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            add_prefix_space=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            config=self.config,
        )
        model.resize_token_embeddings(len(self.tokenizer))

        self.trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
        )

    def process_instances(self, instances: Iterable[str]) -> List[str]:
        def preprocess(_instances):
            tokenized_inputs = self.tokenizer(
                [instance.split() for instance in _instances["text"]],
                padding=True,
                max_length=512,
                truncation=True,
                is_split_into_words=True,
            )
            return tokenized_inputs

        raw_dataset = Dataset.from_dict({"text": instances})

        tokenized_dataset = raw_dataset.map(preprocess)
        label_dict = getattr(self.config, "id2label")

        raw_predictions, _, _ = self.trainer.predict(tokenized_dataset)
        predictions = np.argmax(raw_predictions, axis=1)
        return [label_dict[label_index] for label_index in predictions]


class TaggingModelInterface(ModelInterface):
    def load_from_path(self, path: str) -> None:
        self.label_list = ["O", "B", "I"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # data_collator = DataCollatorForTokenClassification(self.tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(path)
        self.trainer = Trainer(
            model,
            # data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def process_instances(self, instances: Iterable[str]) -> List[str]:
        def preprocess(_instances):
            tokenized_inputs = self.tokenizer(
                [instance.split() for instance in _instances["text"]],
                max_length=128,
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
            )
            tokenized_inputs["token_ids"] = tokenized_inputs.word_ids(batch_index=0)
            return tokenized_inputs

        raw_dataset = Dataset.from_dict({"text": instances})
        tokenized_dataset = raw_dataset.map(preprocess)
        raw_predictions = self.trainer.predict(tokenized_dataset).predictions
        predictions = np.argmax(raw_predictions, axis=2)

        def group_to_label(group: Iterable[Tuple[int, int]]) -> str:
            _, subword_tags = unzip(group)
            first_label = next(subword_tags)
            return self.label_list[first_label]

        def get_tags(index: int, prediction: Iterable[int]) -> str:
            relevant_token_ids_and_tags = (
                (token_id, tag)
                for token_id, tag in zip(
                    tokenized_dataset[index]["token_ids"], prediction
                )
                if token_id is not None
            )
            return "".join(
                group_to_label(g)
                for _, g in groupby(relevant_token_ids_and_tags, lambda s: s[0])
            )

        return [
            get_tags(index, prediction) for index, prediction in enumerate(predictions)
        ]
