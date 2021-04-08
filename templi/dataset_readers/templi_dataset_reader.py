import copy
import gzip
import json
import logging
import os
import tarfile
from typing import Any, Dict, Iterable, List
import numpy as np
from allennlp.data import DatasetReader
from allennlp.data.fields import (
    Field,
    IndexField,
    ListField,
    MetadataField,
    TextField,
    ArrayField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from allennlp_semparse.common import ParsingError
from allennlp_semparse.fields import KnowledgeGraphField, ProductionRuleField
from overrides import overrides
from templi.preprocessors.temli_tokenizer import TemliTokenizer
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext

logger = logging.getLogger(__name__)
max_seq_length = (
    128  # TODO this value default from cosmosqa, change this to a better way
)


@DatasetReader.register("templi")
class TempliDatasetReader(DatasetReader):
    def __init__(self, training, bert_model, do_lower_case, lazy=False, predicting=False) -> None:
        super().__init__(lazy=lazy)
        self.sentences_logical_forms = None  # type: Dict[str:Dict[str:Any]]
        self._tokenizer = TemliTokenizer(bert_model, do_lower_case=do_lower_case)
        # we force SingleIdTokenIndexer to use namespace=None, feature_name="text_id" so
        # the TextField would return Tensor that represents id
        self._question_token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace=None, feature_name="text_id")
        }
        self._table_token_indexers = self._question_token_indexers
        self._training = training
        self._predicting= predicting
        pass

    """
    This function overrides Allennlp's dataset reader
    Returns an Iterator if Instances
    The input dataset must look like this:
    {
        "sentences":{
            "The financial assistance from the World Bank and the International Monetary Fund are not helping.": {
                "75_82": {                  // the "me_event"
                    "target_relations": {   // all the events in the key
                        "12_22": "P"        // what is "me_event" related to "that_event"?
                    },
                    "logical_forms": [
                        "(P 12_22)",        // valid logical form that evaluates correct relations in target_relations
                    ]
                },...
            },...
        "knowledge_graph":{
            "12/25":["Christmans",...],
            "1/1":["New Years Day",...],
            ...
        }
    }
    Note that 12_22 means key_without_spaces[12:22]
    """

    def _read(self, file_path: str) -> Iterable[Instance]:
        if self.sentences_logical_forms is None:
            with open(file_path, "r") as f:
                data = json.load(f)
                self.sentences_logical_forms = {}
                for sentence, v in data["sentences"].items():
                    self.sentences_logical_forms[sentence] = []
                    for main_var, answer in v.items():
                        # this is one data that is going to be tokenized and then converted to Instance
                        one_data = {
                            "sentence": sentence,
                            "main_var": main_var,
                            "temp_vars": list(v.keys()),
                            "target_relations": answer["target_relations"],
                            "logical_forms": answer["logical_forms"],
                            "knowledge_graph": data["knowledge_graph"],
                        }
                        self.sentences_logical_forms[sentence].append(one_data)

        if self._training:
            return filter(
                None.__ne__,
                map(
                    lambda x: self.text_to_instance(x),
                    sum(self.sentences_logical_forms.values(), []),
                ),
            )
        else:
            def merge_instances(instances):
                instances = [i for i in instances if i]
                if not instances: return None

                # note that all instances for a data use the exactly same TemliLanguage instance. This
                # saves the headache later when we execute the logical forms, as the id of intervalvars would
                # be exactly the same. THIS IS NOT THE CASE DURING TRAINING
                fields = instances[0].fields
                
                # TODO currently, seems like only question field and metadata field contains me_event-specific data
                # Things may change in the future
                # TODO variable length ListField would cause
                # Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
                # error. We pad them with to len==10 with last question instead (THIS IS NASTY FIX THIS)
                list_of_questions = [i.fields['question'] for i in instances[:10]]
                list_of_questions += [instances[-1].fields['question']]*(10-len(list_of_questions))
                fields['question'] = ListField(list_of_questions)
                list_of_metadata = [i.fields['metadata'] for i in instances[:10]]
                list_of_metadata += [instances[-1].fields['metadata']]*(10-len(list_of_metadata))
                fields['metadata'] = ListField(list_of_metadata)
                return Instance(fields)

            return filter(
                None.__ne__,
                map(
                    lambda x: merge_instances([self.text_to_instance(i) for i in x]),
                    self.sentences_logical_forms.values(),
                ),
            )

    """
    This function converts one data to an Instance (which contains Dict[str,Field])
    """

    def text_to_instance(self, one_data: Dict[str, Any]) -> Instance:
        # Question Field contains the problem specification (actually just tokenized one_data)
        # This is exactly the input that is going to feed into BertMultiwayMatch's forward()
        # except `doc_len` and `ques_len` (stored in the metadata_field because they aren't converted to tensors)
        # This part is adapted from run_multiway_att.py
        sentence_tokens = self._tokenizer.tokenize_to_str(one_data["sentence"].lower())
        me_event_tokens = self._tokenizer.tokenize_to_str(
            self.spaceless_rng_to_str(
                one_data["sentence"], one_data["main_var"]
            ).lower()
        )
        ques_len = len(me_event_tokens)
        # Modifies `sentence_tokens` and `me_event_tokens` in place so that the total length is less than the
        # specified length.  Account for [CLS], [SEP], [SEP] with "- 3"
        self._truncate_seq_pair(sentence_tokens, me_event_tokens, max_seq_length - 3)
        doc_len = len(sentence_tokens)
        tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"] + me_event_tokens + ["[SEP]"]
        segment_ids = [0] * (len(sentence_tokens) + 2) + [1] * (
                len(me_event_tokens) + 1
        )
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        question_field = ArrayField(
            np.array([input_ids, segment_ids, input_mask]), dtype=np.long
        )

        # Word Field contains the language used by this data
        context = TempliTimeContext(
            temp_vars={
                var: self.spaceless_rng_to_str(one_data["sentence"], var)
                for var in one_data["temp_vars"]
            },
            knowledge_graph=one_data["knowledge_graph"],
        )
        world = TempliLanguage(context)
        world_field = MetadataField(world)

        # Table Field contains the knowledge graph given the context
        table_field = KnowledgeGraphField(
            context.get_table_knowledge_graph(),
            self._tokenizer.tokenize(
                one_data["sentence"].lower()
            ),  # TODO put entire question including me_event or just sentence?
            self._table_token_indexers,
            tokenizer=self._tokenizer,
        )

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
                if not self._predicting:
                    return None
                target_action_sequences_field = None
            else:
                target_action_sequences_field = ListField(action_sequence_fields)
        else:
            if not self._predicting:
                return None
            target_action_sequences_field = None

        # Metadata Field containes the original data (unmodified) plus information required
        # to run BertMultiwayAttention
        metadata_field = MetadataField(
            {**one_data, **{"doc_len": doc_len, "ques_len": ques_len}}
        )

        fields = {
            "question": question_field,
            "table": table_field,
            "world": world_field,
            "actions": action_field,
            "metadata": metadata_field,
        }
        if target_action_sequences_field:
            fields["target_action_sequences"]= target_action_sequences_field

        return Instance(fields)

    """
    This function extracts the substring from a sentence using a range string "xx_yy:
    """

    def spaceless_rng_to_str(self, sentence: str, spacelspaceless_rng: str):
        range_tuple = tuple(map(lambda x: int(x), spacelspaceless_rng[1:-1].split(":")))
        return sentence.replace(" ", "")[range_tuple[0]: range_tuple[1]]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
