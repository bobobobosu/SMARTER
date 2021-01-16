import os
from tqdm import tqdm
from pathlib import Path
import sys
import json
import glob
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
    "MAX_LF_NUM": 20000,
    "DPD_THREADS": 12,  # doesn't affect outcome
    "KG_VERTICES": 5,
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
    from templi.dataset_converters.get_temporal_subset import get_temporal_subset

    def folder_of_tml_to_sentence_rels(folderpath):
        file_list = []
        for filename in glob.iglob(folderpath + '/**/*.tml', recursive=True):
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
                sentence_rels = {**sentence_rels, **parser(news)}
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

    traing_data_path = "training/data/timebank_1_2/data/extra"
    traing_data["sentences"] = sentence_rels_to_temli_data(
        folder_of_tml_to_sentence_rels(traing_data_path)
    )

    validation_data_path = "training/data/timebank_1_2/data/timeml"
    validation_data["sentences"] = sentence_rels_to_temli_data(
        folder_of_tml_to_sentence_rels(validation_data_path)
    )

    # balance uneven data
    for k in list(validation_data["sentences"].keys())[200:]:
        traing_data["sentences"][k] = validation_data["sentences"][k]
        del validation_data["sentences"][k]




# Writes to file
Path("training/cache").mkdir(parents=True, exist_ok=True)
json.dump(
    traing_data,
    open("training/cache/training_data.json", "w"),
    indent=4,
)
json.dump(
    validation_data,
    open("training/cache/validation_data.json", "w"),
    indent=4,
)
