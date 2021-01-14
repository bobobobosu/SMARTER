import pathlib
import shutil
import json
import os
import sys
import time
from argparse import ArgumentParser
from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params
from allennlp.data import Batch, DatasetReader, Vocabulary
from allennlp.models import Model, load_archive
from allennlp.predictors import Predictor

"""
This file is used to train the model given the cached dataset is present
Pass in --recover if you want to recover from previous progress
All file in training/cache folder must be in temli format
"""

sys.path.append(os.getcwd())  # add project root to path

from templi.dataset_readers.templi_dataset_reader import TempliDatasetReader
from templi.models.templi_semantic_parser import TempliSemanticParser
from templi.models.templi_mml_semantic_parser import (
    TempliMmlSemanticParser,
    TempliSemanticParser,
)

trainfilepth = "training/cache/training_data.json"
valfilepth = "training/cache/validation_data.json"
serializationpath = "training/serialization"

p = Params(
    {
        "dataset_reader": {
            "type": "templi.dataset_readers.templi_dataset_reader.TempliDatasetReader",
            "training": True,
            "lazy": True,
            "bert_model": "bert-base-uncased",
            "do_lower_case": True,
        },
        "validation_dataset_reader": {
            "type": "templi.dataset_readers.templi_dataset_reader.TempliDatasetReader",
            "training": False,
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


parser = ArgumentParser()
parser.add_argument("--recover", help="recover?", required=False, action="store_true")
args = parser.parse_args()
if args.recover:
    train_model(
        params=Params.from_file(f"{serializationpath}/config.json"),
        serialization_dir=serializationpath,
        file_friendly_logging=False,
        recover=True,
        force=False,
        node_rank=0,
        include_package=None,
        dry_run=False,
    )
else:
    try:
        os.rename(
            serializationpath, f"{os.path.dirname(serializationpath)}/{time.time()}"
        )
    except:
        pass
    train_model(
        params=p,
        serialization_dir=serializationpath,
        file_friendly_logging=False,
        recover=False,
        force=False,
        node_rank=0,
        include_package=None,
        dry_run=False,
    )
