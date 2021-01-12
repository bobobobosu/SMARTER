import json
from tqdm import tqdm
from templi.templi_languages.allen_algebra import converse, timeml_to_uci
from typing import Dict, Tuple
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext

from allennlp_semparse.common.action_space_walker import ActionSpaceWalker
from functools import reduce
from multiprocessing import Pool


def get_valid_logical_forms(sentences_rels: Dict[str, Dict[str, str]], params={}):
    params["DPD_THREADS"] = 4 if not "DPD_THREADS" in params else params["DPD_THREADS"]

    with Pool(params["DPD_THREADS"]) as p:
        data = list(
            tqdm(
                p.imap(
                    logical_forms_of_sentence,
                    [(k, v, params) for k, v in sentences_rels.items()],
                ),
                total=len(sentences_rels),
            )
        )
        p.close()
        sentences_logical_forms = reduce(lambda a, b: {**a, **b}, data, {})
        return sentences_logical_forms


def logical_forms_of_sentence(sentence_rels: Tuple):
    sentence = sentence_rels[0]
    rels = sentence_rels[1]
    params = sentence_rels[2]
    params["LF_LEN"] = 8 if not "LF_LEN" in params else params["LF_LEN"]
    params["MAX_LF_NUM"] = 5000 if not "MAX_LF_NUM" in params else params["MAX_LF_NUM"]

    if not rels:
        return {}

    # convert pos to str, timeml to uci
    for idx, rel in enumerate(rels):
        for pos_key in {"lhs", "rhs"}:
            rels[idx][pos_key] = f"{rel[pos_key][0]}_{rel[pos_key][1]}"
        rels[idx]["rel"] = timeml_to_uci(rels[idx]["rel"])

    # populate temp_vars
    temp_vars = set(sum([[rel["lhs"], rel["rhs"]] for rel in rels], []))
    # initialize {main_var:logical_forms}
    result = {i: [] for i in temp_vars}

    """
    we create logical form for one variable once at a time, so the context is all variables
    except the main variable (the one logical form should evaluate to)
    """
    for main_var in temp_vars:
        # generate possible logical forms
        target_vars = temp_vars.difference({main_var})
        context = TempliTimeContext(temp_vars=target_vars)
        world = TempliLanguage(context)
        walker = ActionSpaceWalker(world, max_path_length=params["LF_LEN"])
        all_logical_forms = walker.get_all_logical_forms(
            max_num_logical_forms=params["MAX_LF_NUM"]
        )

        # generate target relations
        target_relations = {}  # {target_var: rel}
        for rel in rels:
            if rel["lhs"] == rel["rhs"]:
                # TODO don't know why timeml_parser.py produces this... might be a bug
                continue
            if main_var == rel["lhs"]:
                target_relations[rel["rhs"]] = rel["rel"]
            if main_var == rel["rhs"] and converse(rel["rel"]):
                target_relations[rel["lhs"]] = converse(rel["rel"])

        # filter correct logical forms
        correct_logical_forms = []
        for logical_form in all_logical_forms:
            if world.evaluate_logical_form(logical_form, target_relations):
                correct_logical_forms.append(logical_form)

        # collect training data
        result[main_var] = {
            "target_relations": target_relations,
            "logical_forms": correct_logical_forms,
        }
    return {sentence: result}
