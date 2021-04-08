from overrides import overrides
import torch
import sys
import os

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from templi.dataset_readers.templi_dataset_reader import TempliDatasetReader
from templi.dataset_converters.timeml_parser import parser
from templi.models.temli_multiway_match import (
    BertHierarchicalEventAnnotator,
    BertTernaryEventAnnotator,
)
from bert_event_annotator import annotate_one_instance


@Predictor.register("temli-parser")
class TemliParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.wikitables_semantic_parser.WikiTablesSemanticParser`
    model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "events": "xxxx"}``.
        """
        sentence = json_dict["question"]

        # extract events from question_text
        config_tag = "-bert-base-uncased-hier(torch)-eps_00"
        data_paths = ["timeml", "extra"]
        model_out_dir = (
            "training/event-annotator" + config_tag
        )

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("\n" + "#" * 20 + " " + config_tag + " " + "#" * 20)

        # Load model
        model = torch.load(os.path.join(model_out_dir, "model.pt"))

        # Annotate
        annotations = annotate_one_instance(model, tokenizer, sentence)

        # Visualize annotations
        sentence_no_spacing = sentence.replace(" ", "")
        annotated_events = [sentence_no_spacing[s:e] for s, e in annotations]
        print(annotated_events)

        # extract time from question text


        temp_vars = [f"{i[0]}_{i[1]}" for i in annotations]
        reader = TempliDatasetReader(training=False, bert_model="bert-base-uncased",do_lower_case= True, predicting=True)
        reader.sentences_logical_forms = {}
        reader.sentences_logical_forms[sentence] = [{
                            "sentence": sentence,
                            "main_var": e,
                            "temp_vars": temp_vars,
                            "target_relations": {},
                            "logical_forms": [],
                            "knowledge_graph": {},
                        } for e in temp_vars]

        instances = list(reader._read(file_path=None))
        return instances

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        We need to override this because of the interactive beam search aspects.
        """
        
        instance = self._json_to_instance(inputs)[0]
        self._model.vocab.extend_from_instances([instance])

        result = self.predict_instance(instance)
        return result

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return outputs['holistic_denotation']