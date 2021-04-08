import json
from tqdm import tqdm
from templi.templi_languages.allen_algebra import converse, timeml_to_uci
from typing import Dict, Tuple
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext
from templi.dataset_converters.gen_logical_form_templates import LogicalFormTemplates
from allennlp_semparse.common.action_space_walker import ActionSpaceWalker
from functools import reduce
import copy
from multiprocessing import Pool, Process, Manager
import os
from pathlib import Path

template = None
def get_valid_logical_forms(
    sentences_rels: Dict[str, Dict[str, str]], logical_form_templates=None, params={}
):
    params["DPD_THREADS"] = 4 if not "DPD_THREADS" in params else params["DPD_THREADS"]
    global template
    template = LogicalFormTemplates(params=params)

    # single threaded version for debug
    # data = []
    # for k, v in tqdm(sentences_rels.items()):
    #     data += [logical_forms_of_sentence((k, v, params))]
    # sentences_logical_forms = reduce(lambda a, b: {**a, **b}, data, {})
    # return sentences_logical_forms

    # multithreaded version
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
    global template
    params["LF_LEN"] = 8 if not "LF_LEN" in params else params["LF_LEN"]
    params["MAX_LF_NUM"] = 5000 if not "MAX_LF_NUM" in params else params["MAX_LF_NUM"]

    if not rels:
        return {}

    # convert pos to str, timeml to uci
    for idx, rel in enumerate(rels):
        for pos_key in {"lhs", "rhs"}:
            rels[idx][pos_key] = f"[{rel[pos_key][0]}:{rel[pos_key][1]}]"
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

        target_vars = temp_vars.difference({main_var})
        context = TempliTimeContext(
            temp_vars=target_vars, knowledge_graph={}
        )  # knowledge_graph not required to generate lf
        world = TempliLanguage(context)


        all_logical_forms = []
        # hard coded lf that is always valid
        ub_lf, ub_len = upperbound_lf(main_var, target_relations)
        if ub_lf:
            all_logical_forms += [ub_lf]

        if len(target_relations) > 2:
            rrr = 8

        hard_coded_lf_len = {
            1: 6,
            2: 9,
            3: 11
        }
        # params["LF_LEN"] = hard_coded_lf_len[len(target_relations)] if len(target_relations) in hard_coded_lf_len else 1

        if len(target_relations) == 1:
            Path("training/cache/memo").mkdir(parents=True, exist_ok=True)

            tempvar_idx = {k: f"${i}$" for i, k in enumerate(list(target_vars))}
            fingerprint = copy.deepcopy(target_relations)
            fingerprint = {k: fingerprint[k] if k in  fingerprint else '' for k in target_vars}
            fingerprint = {tempvar_idx[k]: v for k, v in fingerprint.items()}
            fingerprint = sorted([(k, v) for k, v in fingerprint.items()], key=lambda x:x[0])

            filepath = f"training/cache/memo/{str(fingerprint)}.json"
            if not os.path.isfile(filepath):
                walker = ActionSpaceWalker(world, params=params)
                all_logical_forms += [i for i in walker.iterate_all_logical_forms()]

                # filter correct logical forms
                correct_logical_forms = set()
                for logical_form in all_logical_forms:
                    action_sequence = world.logical_form_to_action_sequence(logical_form)
                    if params["LF_PARTIAL_MATCH"]:
                        if world.evaluate_logical_form_partial_match(logical_form, target_relations) > 0:
                            correct_logical_forms.add(logical_form)
                    else:
                        if world.evaluate_logical_form(logical_form, target_relations):
                            correct_logical_forms.add(logical_form)

                # serialize template
                template_logical_forms = copy.deepcopy(correct_logical_forms)
                for k,v in tempvar_idx.items():
                    template_logical_forms = [x.replace(k, v) for x in template_logical_forms]
                json.dump(
                    template_logical_forms,
                    open(filepath, "w"),
                    indent=4,
                )
            else:
                # deserializa template
                with open(filepath, "r") as data_file:
                    correct_logical_forms = json.load(data_file)
                for k,v in tempvar_idx.items():
                    correct_logical_forms = [x.replace(v, k) for x in correct_logical_forms]
            
        else:
            # filter correct logical forms
            correct_logical_forms = set()
            for logical_form in all_logical_forms:
                action_sequence = world.logical_form_to_action_sequence(logical_form)
                if world.evaluate_logical_form(logical_form, target_relations):
                    correct_logical_forms.add(logical_form)

        # collect training data
        result[main_var] = {
            "target_relations": target_relations,
            "logical_forms": list(correct_logical_forms),
        }
    return {sentence: result}


def upperbound_lf(main_var, target_relations):
    length = 0
    each_rel = []
    for k, v in target_relations.items():
        each_rel += [f"(op1_{v} const_{k})"]
    if len(each_rel) == 0:
        return None, None
    if len(each_rel) == 1:
        return each_rel[0], 4

    # TODO increase intersection & union args in gen_templi_language_functions to enable this
    each_rel = each_rel[:3]

    lf = f"(func_intersection_{len(each_rel)} {' '.join(each_rel)})"
    length = len(each_rel)*3
    return lf, length
