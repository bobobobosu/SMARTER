import pathlib
import shutil
import json
import os
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

datapath = 'data'
data_tag = 'small'
data_name = f'sentences_logical_forms_{data_tag}'
datafilenm = f"{datapath}/{data_name}.json"
trainfilepth = f"{datapath}/{data_name}_train.json"
valfilepth = f"{datapath}/{data_name}_dev.json"
config_tag = 'hidden_size_10_num_layers_2_small'
split_n = 5

# Generate train & validation files and remove old folders
if not os.path.exists(trainfilepth):
    with open(datafilenm, "r") as f:
        sentences_logical_forms = json.load(f)
    dev_keys = list(sentences_logical_forms.keys())[:len(sentences_logical_forms) // split_n]
    train_keys = list(sentences_logical_forms.keys())[len(sentences_logical_forms) // split_n:]
    json.dump({k: sentences_logical_forms[k] for k in list(dev_keys)}, open(trainfilepth, "w"), indent=4)
    json.dump({k: sentences_logical_forms[k] for k in list(train_keys)}, open(valfilepth, "w"), indent=4)
# try:
#     shutil.rmtree(f"serialization_{config_tag}")
# except:
#     pass

p = Params(
    {
        "dataset_reader": {
            "type": "templi.dataset_readers.templi_dataset_reader.TempliDatasetReader",
            "lazy": True,
            "bert_model": "bert-base-uncased",
            "do_lower_case": True,
        },
        "train_data_path": trainfilepth,
        "validation_data_path": valfilepth,
        "model": {
            "type": "templi.models.templi_mml_semantic_parser.TempliMmlSemanticParser",
            "bert_model": "bert-base-uncased",
            "action_embedding_dim": 50,
            "encoder": {
                "type": "lstm",
                "input_size": 1536,
                "hidden_size": 10,
                "num_layers": 2,
            },
            "entity_encoder": {"type": "boe", "embedding_dim": 768, "averaged": True},
            "decoder_beam_search": {"beam_size": 3},
            "max_decoding_steps": 200,
            "attention": {"type": "dot_product"},
            "cuda_device": 0,
            "num_linking_features": 0,  # TODO implement linking features
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
    params=Params.from_file('serialization/config.json'),
    serialization_dir='serialization',
    # serialization_dir=f"serialization_{config_tag}",
    file_friendly_logging=False,
    recover=True,
    force=False,
    node_rank=0,
    include_package=None,
    dry_run=False,
)
