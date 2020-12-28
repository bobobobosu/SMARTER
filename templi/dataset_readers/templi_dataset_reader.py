import logging
from typing import Dict, List, Any, Iterable
import os
import gzip
import tarfile
import json
import copy
from overrides import overrides

from allennlp.data import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext

from allennlp_semparse.common import ParsingError
from allennlp_semparse.fields import KnowledgeGraphField, ProductionRuleField

from templi.dataset_readers.tokenization import BertTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("templi")
class TempliDatasetReader(DatasetReader):
    def __init__(self, bert_model, do_lower_case, lazy=False) -> None:
        super().__init__(lazy=lazy)
        self.sentences_logical_forms = None  # type: Dict[str:Dict[str:Any]]
        self._tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=do_lower_case
        )
        pass

    """
    This function overrides Allennlp's dataset reader
    Returns an Iterator if Instances
    The input dataset must look like this:
    {
    "The financial assistance from the World Bank and the International Monetary Fund are not helping.": {
        "75_82": {                  // the "me_event"
            "target_relations": {   // all the events in the key
                "12_22": "P"        // what is "me_event" related to "that_event"?
            },
            "logical_forms": [
                "(P 12_22)",        // valid logical form that evaluates correct relations in target_relations
            ]
        },...
    }
    Note that 12_22 means key_without_spaces[12:22]
    """

    def _read(self, file_path: str) -> Iterable[Instance]:
        if self.sentences_logical_forms is None:
            with open(file_path, "r") as f:
                data = json.load(f)
                self.sentences_logical_forms = []
                for sentence, v in data.items():
                    for main_var, answer in v.items():
                        # this is one data that is going to be tokenized and then converted to Instance
                        one_data = {
                            "sentence": sentence,
                            "main_var": main_var,
                            "temp_vars": list(v.keys()),
                            "target_relations": answer["target_relations"],
                            "logical_forms": answer["logical_forms"],
                        }
                        self.sentences_logical_forms.append(one_data)
        return filter(
            None.__ne__,
            map(lambda x: self.text_to_instance(x), self.sentences_logical_forms),
        )

    """
    This function converts one data to an Instance (which contains Dict[str,Field])
    """

    def text_to_instance(self, one_data: Dict[str, Any]) -> Instance:

        # Metadata Field containes the original data (unmodified)
        metadata_field = MetadataField(one_data)

        # Question Field contains the problem specification (actually just tokenized one_data)
        question_field = MetadataField(self.tokenize_one_data(one_data))

        # Word Field contains the language used by this data
        context = TempliTimeContext(temp_vars=one_data["temp_vars"])
        world = TempliLanguage(context)
        world_field = MetadataField(world)

        # Production Rule Field contains the production rules possible from this world
        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_productions():
            _, rule_right_side = production_rule.split(" -> ")
            field = ProductionRuleField(
                production_rule, is_global_rule=True
            )  # currently we only have global grammar
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        # Target Action Sequence Field contains possible actions from generated logical forms
        action_map = {
            action.rule: i for i, action in enumerate(action_field.field_list)
        }  # type: ignore
        # We'll make each target action sequence a List[IndexField], where the index is into
        # the action list we made above.  We need to ignore the type here because mypy doesn't
        # like `action.rule` - it's hard to tell mypy that the ListField is made up of
        # ProductionRuleFields.
        if one_data["logical_forms"]:
            action_sequence_fields: List[Field] = []
            for logical_form in one_data["logical_forms"]:
                try:
                    action_sequence = world.logical_form_to_action_sequence(
                        logical_form
                    )
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(
                            IndexField(action_map[production_rule], action_field)
                        )
                    action_sequence_fields.append(ListField(index_fields))
                except:  # noqa
                    logger.error(logical_form)
                    raise

            if not action_sequence_fields:
                # This is not great, but we're only doing it when we're passed logical form
                # supervision, so we're expecting labeled logical forms, but we can't actually
                # produce the logical forms.  We should skip this instance.  Note that this affects
                # _dev_ and _test_ instances, too, so your metrics could be over-estimates on the
                # full test data.
                return None
            target_action_sequences_field = ListField(action_sequence_fields)
        else:
            return None

        fields = {
            "question": question_field,
            "world": world_field,
            "actions": action_field,
            "metadata": metadata_field,
            "target_action_sequences": target_action_sequences_field,
        }

        return Instance(fields)

    """
    This function tokenizes the data if needed
    """

    def tokenize_one_data(self, one_data):
        one_data = copy.deepcopy(one_data)

        # helper for tokenization
        def tokenize_to_ids(sentence: str):
            return self._tokenizer.convert_tokens_to_ids(
                self._tokenizer.tokenize(sentence)
            )

        # tokenize main_var
        one_data["main_var"] = tokenize_to_ids(
            self.spaceless_rng_to_str(one_data["sentence"], one_data["main_var"])
        )

        # tokenize temp_vars
        one_data["temp_vars"] = [
            tokenize_to_ids(self.spaceless_rng_to_str(one_data["sentence"], i))
            for i in one_data["temp_vars"]
        ]

        # tokenize sentence
        one_data["sentence"] = tokenize_to_ids(one_data["sentence"])

        return one_data

    """
    This function extracts the substring from a sentence using a range string "xx_yy:
    """

    def spaceless_rng_to_str(self, sentence: str, spacelspaceless_rng: str):
        range_tuple = tuple(map(lambda x: int(x), spacelspaceless_rng.split("_")))
        return sentence.replace(" ", "")[range_tuple[0] : range_tuple[1]]
