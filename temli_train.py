import pathlib
import shutil

from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params
from allennlp.data import Batch, DatasetReader, Vocabulary
from allennlp.models import Model, load_archive
from allennlp.predictors import Predictor

from templi.dataset_readers.templi_dataset_reader import TempliDatasetReader
from templi.models.templi_semantic_parser import TempliSemanticParser
from templi.models.templi_mml_semantic_parser import (
    TempliMmlSemanticParser,
    TempliSemanticParser,
)

# from allennlp_semparse.predictors.wikitables_parser import WikiTablesParserPredictor
# from allennlp_semparse_pretrained import _load_predictor, wikitables_parser_dasigi_2019
# from allennlp.predictors.predictor import Predictor

try:
    shutil.rmtree("serialization")
except:
    pass

p = Params(
    {
        "dataset_reader": {
            "type": "templi.dataset_readers.templi_dataset_reader.TempliDatasetReader",
            "lazy": True,
            "bert_model": "bert-base-uncased",
            "do_lower_case": True,
        },
        "train_data_path": "/mnt/AAI_Project/temli/data/sentences_logical_forms_train.json",
        "validation_data_path": "/mnt/AAI_Project/temli/data/sentences_logical_forms_dev.json",
        "model": {
            "type": "templi.models.templi_mml_semantic_parser.TempliMmlSemanticParser",
            "bert_model": "bert-base-uncased",
            "action_embedding_dim": 50,
            "encoder": {
                "type": "lstm",
                "input_size": 1536,
                "hidden_size": 10,
                "num_layers": 1,
            },
            "entity_encoder": {"type": "boe", "embedding_dim": 768, "averaged": True},
            "decoder_beam_search": {"beam_size": 3},
            "max_decoding_steps": 200,
            "attention": {"type": "dot_product"},
            "cuda_device": 0,
            "num_linking_features": 0, # TODO implement linking features
        },
        "data_loader": {"batch_size": 16},
        "trainer": {
            "num_epochs": 10000,
            "patience": 10,
            "cuda_device": 0,
            "optimizer": {"type": "sgd", "lr": 0.01},
        },
    }
)
train_model(
    params=p,
    serialization_dir="serialization",
    file_friendly_logging=False,
    recover=False,
    force=False,
    node_rank=0,
    include_package=None,
    dry_run=False,
)