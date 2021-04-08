import os
from tqdm import tqdm
from pathlib import Path
from functools import reduce
import sys
import json
import glob
import gc
import copy

"""
This file converts raw data file to temli format
You have to edit this manually to call the right converters
Our goal here is to produce `training/cache/traing_data.json` and 
`training/cache/validation_data.json`
THIS FILE HAS TO BE RUN FROM PROJECT ROOT
"""

sys.path.append(os.getcwd())  # add project root to path

# Tunable Params
cache_params = {
    "LF_LEN": 8,
    "MAX_LF_NUM": 300000,
    "LF_PARTIAL_MATCH": False,
    "DPD_THREADS": 10,  # doesn't affect outcome
    "KG_VERTICES": 5,
    "SENTENCES_ONE_DATA": 1,
    "LF_TEMPLATE_PATH": "training/cache/logical_form_templates.json",
    "LF_TEMPLATE_CACHE_LEN": 1000000,
    "LF_RUNTIME_CACHE_LEN": 20000000,
    "LF_RUNTIME_CACHE_PATH": "training/cache/logical_form_runtime_cache.json",
}


#  This is the converter template for TimeML datasets
traing_data = validation_data = {"sentences": {}, "knowledge_graph": {}}
manual_split = False
if manual_split:
    # if the dataset have to be splitted manually
    pass
else:
    # if the dataset is already splitted to training and validation
    from templi.dataset_converters.timeml_parser import parser
    from templi.dataset_converters.search_timeml_logical_forms import (
        get_valid_logical_forms,
    )
    from templi.dataset_converters.gen_logical_form_templates import (
        LogicalFormTemplates,
    )
    from templi.dataset_converters.get_temporal_subset import get_temporal_subset
    from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext
    from allennlp_semparse.common.action_space_walker import ActionSpaceWalker

    def folder_of_tml_to_sentence_rels(folderpath):
        file_list = []
        for filename in glob.iglob(folderpath + "/**/*.tml", recursive=True):
            file_list += [filename]
        print("parse {}...".format(folderpath))
        sentence_rels = {}
        for file in tqdm(file_list, desc="extracting sentence_rels"):
            with open(file) as f:
                news = f.read().replace("\n", "")
                news = news.replace(
                    "encoding='us-ascii'", ""
                )  # hack to make timeml-dense work
                news = news.replace("<TEXT>", "")  # hack to make timeml-dense work
                news = news.replace("</TEXT>", "")  # hack to make timeml-dense work
                sentence_rels = {
                    **sentence_rels,
                    **parser(
                        news, sentences_in_one_data=cache_params["SENTENCES_ONE_DATA"]
                    ),
                }
            # if len(sentence_rels) > 6:
            #     break
        return sentence_rels

    def sentence_rels_to_temli_data(sentence_rels):
        sentence_rels = {
            k: [i for i in v if i["rel"] != "NONE"] for k, v in sentence_rels.items()
        }  # hack to make timeml-dense work

        temli_data = get_valid_logical_forms(sentence_rels, params=cache_params)
        return temli_data

    def tables_to_knowledge_graph(knowledge_graph_data_path):
        list_of_tables = get_temporal_subset(
            knowledge_graph_data_path, ratio_threshold=0.1
        )
        kg = {}
        for table in list_of_tables:
            for column in table.columns:
                for value in table[column]:
                    column = str(column)
                    value = str(value)
                    kg[column] = [value] if column not in kg else kg[column] + [value]
                    kg[value] = [column] if value not in kg else kg[value] + [column]
        kg = {k: set(v) for k, v in kg.items()}

        # parameterize KG size
        vertices_to_drop = set(list(kg.keys())[cache_params["KG_VERTICES"] :])
        for v in vertices_to_drop:
            del kg[v]
        kg = {k: v.difference(vertices_to_drop) for k, v in kg.items()}

        # make serializable
        return {k: list(v) for k, v in kg.items()}

    Path("training/cache").mkdir(parents=True, exist_ok=True)

    # Generate knowledge_graph data
    # knowledge_graph_data_path = "training/data/WikiTableQuestions"
    # traing_data["knowledge_graph"] = validation_data[
    #     "knowledge_graph"
    # ] = tables_to_knowledge_graph(knowledge_graph_data_path)

    # Generate sentences data
    # traing_data_path = "training/data/tbaq-2013-03"
    # traing_data["sentences"] = sentence_rels_to_temli_data(
    #     folder_of_tml_to_sentence_rels(traing_data_path)
    # )

    # validation_data_path = "training/data/te3-platinumstandard"
    # validation_data["sentences"] = sentence_rels_to_temli_data(
    #     folder_of_tml_to_sentence_rels(validation_data_path)
    # )


    # # This segmant of code removes invalid logical forms from training data
    # with open("training/cache/training_data.json", "r") as data_file:
    #     traing_data = json.load(data_file)

    # with open("training/cache/training_data.json", "r") as data_file:
    #     validation_data = json.load(data_file)

    # all_sentence_data = {**traing_data["sentences"], **validation_data["sentences"]}
    # for k, v in tqdm(copy.deepcopy(all_sentence_data).items()):
    #     for main_var, data in v.items():
    #         context = TempliTimeContext(
    #             temp_vars=set(v.keys()), knowledge_graph={}
    #         )  # knowledge_graph not required to generate lf
    #         world = TempliLanguage(context)
    #         good_logical_forms = []
    #         for lf in copy.deepcopy(data["logical_forms"]):
    #             if world.evaluate_logical_form(lf, data["target_relations"]):
    #                 good_logical_forms.append(lf)
    #         all_sentence_data[k][main_var]["logical_forms"] = good_logical_forms
    # traing_data["sentences"] = all_sentence_data
    # validation_data["sentences"] = all_sentence_data
    # json.dump(
    #     traing_data,
    #     open("training/cache/training_data.json", "w"),
    #     indent=4,
    # )    
    # json.dump(
    #     validation_data,
    #     open("training/cache/validation_data.json", "w"),
    #     indent=4,
    # )
    # exit()



    # # optionally generate templates for lf
    # template = LogicalFormTemplates(params=cache_params)
    # logical_form_templates = {}
    # file_cnter = 0
    # for lf_lflen in tqdm(template.iterate_logical_form_templates(), desc="Generating lf templates"):
    #     for k, v in lf_lflen.items():
    #         if k not in logical_form_templates:
    #             logical_form_templates[k] = []
    #         logical_form_templates[k] += v
    #     # if len(logical_form_templates) > cache_params["LF_TEMPLATE_CACHE_LEN"]:
    #     #     # Writes to file
    #     #     file_cnter += 1
    #     #     Path("training/cache").mkdir(parents=True, exist_ok=True)
    #     #     file_path = cache_params["LF_TEMPLATE_PATH"]
    #     #     json.dump(
    #     #         logical_form_templates,
    #     #         open(f"{file_path}.{str(file_cnter)}", "w"),
    #     #         indent=4,
    #     #     )
    #     #     logical_form_templates = []
    #     #     gc.collect()
    # json.dump(
    #     logical_form_templates,
    #     open(cache_params["LF_TEMPLATE_PATH"], "w"),
    #     indent=4,
    # )

    # exit()

    traing_data_path = "training/data/timebank_1_2/data/all"
    traing_data["sentences"] = sentence_rels_to_temli_data(
        folder_of_tml_to_sentence_rels(traing_data_path)
    )

    json.dump(
        traing_data,
        open("training/cache/training_data.json", "w"),
        indent=4,
    )

    # validation_data_path = "training/data/timebank_1_2/data/timeml"
    # validation_data["sentences"] = sentence_rels_to_temli_data(
    #     folder_of_tml_to_sentence_rels(validation_data_path)
    # )
    json.dump(
        traing_data,
        open("training/cache/validation_data.json", "w"),
        indent=4,
    )


## 10-fold cross validation
# load data

with open("training/cache/training_data.json", "r") as data_file:
    traing_data = json.load(data_file)

with open("training/cache/validation_data.json", "r") as data_file:
    validation_data = json.load(data_file)

all_sentence_data = {**traing_data["sentences"], **validation_data["sentences"]}


def chunks(lst, n):
    lst = list(lst)
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


ten_sets = list(chunks(all_sentence_data.keys(), len(all_sentence_data)//10))

for idx, val in enumerate(ten_sets):
    traing_data = copy.deepcopy(traing_data)
    traing_data["sentences"] = {
        k: all_sentence_data[k]
        for k in sum([x for i, x in enumerate(ten_sets) if i != idx], [])
    }
    validation_data = copy.deepcopy(validation_data)
    validation_data["sentences"] = {k: all_sentence_data[k] for k in val}
    json.dump(
        traing_data,
        open(f"training/cache/training_data_{idx}.json", "w"),
        indent=4,
    )
    json.dump(
        validation_data,
        open(f"training/cache/validation_data_{idx}.json", "w"),
        indent=4,
    )
