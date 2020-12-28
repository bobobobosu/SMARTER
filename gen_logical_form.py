from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import SpacyTokenizer
import timeit
from allennlp.common import Params
from allennlp_semparse.common import Date, ExecutionError
from templi.templi_languages.templi_language import Templi_Language, TempliTimeContext
from templi.dataset_converters.search_timeml_logical_forms import get_all_valid_logical_forms, get_valid_logical_forms
from ortools.linear_solver import pywraplp
from templi.templi_languages.allen_algebra import infer_relation
from allennlp_semparse.common.action_space_walker import ActionSpaceWalker
from allennlp.commands.train import train_model_from_file, train_model
import time
import pathlib
import json


# for generating logical forms
# with open("/mnt/AAI_Project/temli/data/sentence_rels.json", "r") as f:
#     sentence_rels = json.load(f)
# sentences_logical_forms = get_valid_logical_forms(sentence_rels)
# # # write to file
# json.dump(sentences_logical_forms, open("data/sentences_logical_forms.json", "w"), indent=4)

# for generating logical forms (small)
with open("data/sentences_logical_forms.json", "r") as f:
    sentences_logical_forms = json.load(f)
json.dump({k:sentences_logical_forms[k] for k in list(sentences_logical_forms.keys())[:20]}, open("data/sentences_logical_forms_small.json", "w"), indent=4)
fff = 9
