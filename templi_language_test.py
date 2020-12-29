from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import SpacyTokenizer
import timeit
from allennlp.common import Params
from allennlp_semparse.common import Date, ExecutionError
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext
from templi.dataset_converters.search_timeml_logical_forms import get_all_valid_logical_forms, get_valid_logical_forms
from ortools.linear_solver import pywraplp
from templi.templi_languages.allen_algebra import infer_relation
from allennlp_semparse.common.action_space_walker import ActionSpaceWalker
from allennlp.commands.train import train_model_from_file, train_model
import time
import pathlib
import json


# training model
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
# model = train_model()




# for generating logical forms
with open("/mnt/AAI_Project/temli/data/sentence_rels.json", "r") as f:
    sentence_rels = json.load(f)
sentences_logical_forms = get_valid_logical_forms(sentence_rels)
# # write to file
json.dump(sentences_logical_forms, open("sentences_logical_forms_len_8.json", "w"), indent=4)
fff = 9

# for generating training and validation dataset
with open("/mnt/AAI_Project/temli/data/sentences_logical_forms.json", "r") as f:
    sentences_logical_forms = json.load(f)
dev_keys = list(sentences_logical_forms.keys())[:len(sentences_logical_forms)//5]
train_keys = list(sentences_logical_forms.keys())[len(sentences_logical_forms)//5:]
json.dump({k:sentences_logical_forms[k] for k in list(dev_keys)}, open("data/sentences_logical_forms_len_8_dev.json", "w"), indent=4)
json.dump({k:sentences_logical_forms[k] for k in list(train_keys)}, open("data/sentences_logical_forms_len_8_train.json", "w"), indent=4)

# templilanguage = Templi_Language(TempliTimeContext({'clothes_dry','Thursday','evening'}))
# # valid_actions = templilanguage.get_nonterminal_productions()
# # walker = ActionSpaceWalker(templilanguage, max_path_length=5)
# # all_logical_forms = walker.get_all_logical_forms(max_num_logical_forms=1000)
# solver = pywraplp.Solver.CreateSolver("GLOP")

# # logical_form = "(intersection (during (clothes_dry)) (contains (during (intersection (during Thursday) (during evening)))))"
# logical_form = "(intersection (precedes (during (clothes_dry))) (contains (during (intersection (during Thursday) (during evening)))))"
# logical_form = "(intersection (d (clothes_dry)) (D (d (intersection (d Thursday) (d evening)))))"
# final_interval = templilanguage.execute(logical_form)

# using allen algebra

# interval_variables_list = final_interval.get_interval_variables()
# clothes_dry_var = [i for i in interval_variables_list if i.name == "clothes_dry"][0]
# rel = infer_relation(final_interval, final_interval.intervalvar, clothes_dry_var)

# # using solver
# constraints_list = final_interval.constraints_list
# variables_list = final_interval.get_variables()

# for i, constraints in enumerate(constraints_list):
#     id_IntVar_dict = {i.id: solver.NumVar(i.lb, i.ub, str(i.id)) for i in variables_list}
#     for constraint in constraints:
#         ct = solver.Constraint(*constraint.rhs, str(constraint.id))
#         for lhs_tup in constraint.lhs:
#             ct.SetCoefficient(id_IntVar_dict[lhs_tup[0].id], lhs_tup[1])

#     objective = solver.Objective()
#     objective.SetCoefficient(id_IntVar_dict[final_interval.endvar.id], 1)
#     objective.SetCoefficient(id_IntVar_dict[final_interval.startvar.id], -1)
#     objective.SetMinimization()
#     result = solver.Solve()

#     print("Solution:")
#     print("Objective value =", objective.Value())
#     print("start =", id_IntVar_dict[final_interval.startvar.id].solution_value())
#     print("end =", id_IntVar_dict[final_interval.endvar.id].solution_value())
#     print("--- %s seconds ---" % (time.time() - start_time))
#     fff = 9
