import os
from tqdm import tqdm
from pathlib import Path
import sys
import json

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
    "LF_LEN": 1,
    "MAX_LF_NUM": 1,
    "DPD_THREADS": 6,  # don't affect outcome
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
        file_list = os.listdir(folderpath)
        print("parse {}...".format(folderpath))
        sentence_rels = {}
        for file in tqdm(file_list, desc="extracting sentence_rels"):
            with open(os.path.join(folderpath, file)) as f:
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
        list_of_tables = get_temporal_subset(knowledge_graph_data_path, ratio_threshold=0.1)
        kg = {}
        for table in list_of_tables:
            for column in table.columns:
                for value in table[column]:
                    column = str(column)
                    value = str(value)
                    kg[column] = [value] if column not in kg else kg[column] + [value]
                    kg[value] = [column] if value not in kg else kg[value] + [column]
        kg = {k:list(set(v)) for k, v in kg.items()}
        return kg
       
    # Generate knowledge_graph data
    knowledge_graph_data_path = "training/data/WikiTableQuestions"
    traing_data["knowledge_graph"] = validation_data["knowledge_graph"] = tables_to_knowledge_graph(knowledge_graph_data_path)  

    # Generate sentences data
    traing_data_path = "training/data/TimeBank-dense-master/train"
    traing_data["sentences"] = sentence_rels_to_temli_data(
        folder_of_tml_to_sentence_rels(traing_data_path)
    )

    validation_data_path = "training/data/TimeBank-dense-master/test"
    validation_data["sentences"] = sentence_rels_to_temli_data(
        folder_of_tml_to_sentence_rels(validation_data_path)
    )



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
