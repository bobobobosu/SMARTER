from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import SpacyTokenizer
import timeit
from allennlp.common import Params
from allennlp_semparse.common import Date, ExecutionError
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext
from templi.templi_languages.allen_algebra import infer_relation
from allennlp_semparse.common.action_space_walker import ActionSpaceWalker
from allennlp.commands.train import train_model_from_file, train_model
import time
import pathlib
import json



templilanguage = TempliLanguage(TempliTimeContext({"discussed":"discussed","says":"says"},{}))
# # valid_actions = templilanguage.get_nonterminal_productions()
# # walker = ActionSpaceWalker(templilanguage, max_path_length=5)
# # all_logical_forms = walker.get_all_logical_forms(max_num_logical_forms=1000)
# solver = pywraplp.Solver.CreateSolver("GLOP")

logical_form = "(intersection (O discussed) (p says))"
final_interval = templilanguage.execute(logical_form)

ddd = 9