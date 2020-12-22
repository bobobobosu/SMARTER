import pathlib
import shutil

from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params
from allennlp.data import Batch, DatasetReader, Vocabulary
from allennlp.models import Model, load_archive
from allennlp.predictors import Predictor

# from allennlp_semparse.common.wikitables import TableQuestionContext
# from allennlp_semparse.dataset_readers import WikiTablesDatasetReader
# from allennlp_semparse.domain_languages.wikitables_language import WikiTablesLanguage
# from allennlp_semparse.models.wikitables.wikitables_mml_semantic_parser import (
#     WikiTablesMmlSemanticParser,
# )
# from allennlp_semparse.predictors.wikitables_parser import WikiTablesParserPredictor
# from allennlp_semparse_pretrained import _load_predictor, wikitables_parser_dasigi_2019
# from allennlp.predictors.predictor import Predictor

shutil.rmtree('serialization')

p = Params({
    'dataset_reader': 'templi.dataset_readers.templi_dataset_reader.TempliDatasetReader',
    'train_data_path': '/mnt/AAI_Project/temli/data/sentences_logical_forms.json',
    'model': 'templi.models.templi_semantic_parser.TempliSemanticParser',
    'data_loader': 'templi.dataset_readers.templi_data_loader.TempliDataLoader',
    'trainer': {
        "num_epochs": 10000,
        "patience": 10,
        "cuda_device": 0,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01
        }
    }
})
train_model(params=p,
            serialization_dir='serialization',
            file_friendly_logging=False,
            recover=False,
            force=False,
            node_rank=0,
            include_package=None,
            dry_run=False,
            )


# #import allennlp_models.tagging
# #predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
# # fff = predictor.predict(
# #   sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"
# # )
# # import json

# PROJECT_ROOT = (pathlib.Path(__file__).parents[0]).resolve()
# MODULE_ROOT = PROJECT_ROOT / "allennlp_semparse"
# TOOLS_ROOT = None  # just removing the reference from super class
# TESTS_ROOT = PROJECT_ROOT / "tests"
# FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

# print(FIXTURES_ROOT)
# param_file = FIXTURES_ROOT / "wikitables" / "experiment_my.json"
# # dataset_file = FIXTURES_ROOT / "data" / "wikitables" / "sample_data.examples"
# # dataset_file = PROJECT_ROOT / "WikitableQuestions" / "data" / "train.examples"
# params = Params.from_file(param_file)

# ## train from scratch
# # reader = DatasetReader.from_params(params["dataset_reader"])
# model = train_model_from_file(param_file, "./mytrain2", recover=False)
# # model = train_model_from_file(param_file, "/content/drive/My Drive/Research Project_ID_689546573/Lab/temparse/mytrain", recover=False,force=True)
# # vocab = Vocabulary.from_files("./mytrain/vocabulary")
# # # model = Model.from_archive(archive_file="./training_loss_0.9980763257445175.tar.gz")
# # model = Model.from_params(vocab=vocab, params=params["model"])
# # predictor = _load_predictor("./mytrain/model.tar.gz", "wikitables-parser")
# #


# # pretrained
# predictor = wikitables_parser_dasigi_2019()

# with open("WikitableQuestions/csv/203-csv/116.tsv") as csv_file:
#     tableS = csv_file.read().rstrip("\n")
# inputs = {
#     "question": "What is the height of Martti Juhkami",
#     "table": f"{tableS}",
# }
# result = predictor.predict_json(inputs)
# ffo = 0
