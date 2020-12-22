import logging
from typing import Dict, List, Any, Iterable
import os
import gzip
import tarfile
import json
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
from templi.templi_languages.templi_language import Templi_Language, TempliTimeContext


from allennlp_semparse.common import ParsingError
from allennlp_semparse.fields import KnowledgeGraphField, ProductionRuleField


@DatasetReader.register("templi")
class TempliDatasetReader(DatasetReader):
    def __init__(self, lazy=False) -> None:
        super().__init__(lazy=lazy)
        self.flat_sentences_logical_forms = None
        self._tokenizer = SpacyTokenizer(pos_tags=True)
        self._question_token_indexers = {
            "tokens": SingleIdTokenIndexer()
        }
        pass

    def _read(self, file_path: str) -> Iterable[Instance]:
        if self.sentences_logical_forms is None:
            with open(file_path, "r") as f:
                sent_logi = json.load(f)
                self.flat_sentences_logical_forms = sum([[{'sentence': k, 'temp_vars': set(
                    v.keys), 'main_var': main_var, 'logical_form': logical_form} for main_var, logical_form in v.items()] for k, v in sent_logi.items()], [])
        return

    def text_to_instance(self, flat_sentences_logical_form: Dict[Any]) -> Instance:
        sentence = flat_sentences_logical_form['sentence']
        temp_vars = flat_sentences_logical_form['temp_vars']
        main_var = flat_sentences_logical_form['main_var']
        logical_form = flat_sentences_logical_form['logical_form']


        tokenized_question = self._tokenizer.tokenize(sentence.lower())
        question_field = TextField(
            tokenized_question, self._question_token_indexers)

        context = TempliTimeContext(temp_vars=temp_vars)
        world = Templi_Language(context)
        world_field = MetadataField(world)
        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_productions():
            _, rule_right_side = production_rule.split(" -> ")
            field = ProductionRuleField(
                production_rule, is_global_rule=True)  # currently we only have global grammar
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        fields = {
            "sentence": question_field,
            "logical_form":,
            "world": world_field,
            "actions": action_field,
        }

        return Instance(fields)

    def spaceless_rng_to_str(sentence: str, spacelspaceless_rng: str):
        range_tuple = spacelspaceless_rng.split('_')
        return sentence.replace(' ', '')[range_tuple[0]][range_tuple[1]]
