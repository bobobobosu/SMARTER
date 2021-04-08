import sys
import os
from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params
from allennlp.data import Batch, DatasetReader, Vocabulary
from allennlp.models import Model, load_archive
from allennlp.predictors import Predictor
from allennlp.models.archival import Archive, archive_model, load_archive

sys.path.append(os.getcwd())  # add project root to path

from templi.predictors.temli_parser import TemliParserPredictor

archive_model("training/serialization", archive_path="running/serialization.tar.gz")
predictor = Predictor.from_archive(
    load_archive(
        "running/serialization.tar.gz",
        cuda_device=0,
        overrides={"model": {"cuda_device": 0}},
    ),
    "temli-parser",
)
print(
    predictor.predict_json(
        {
            "question": "I didn't think this annotator would work on Jan 14, but it does today.",
        }
    )
)
fff = 9
