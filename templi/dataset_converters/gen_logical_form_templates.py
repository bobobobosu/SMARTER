from typing import Set, Dict, Any
import os
from pathlib import Path
import sys
import json

sys.path.append(os.getcwd())  # add project root to path
from templi.templi_languages.templi_language import TempliLanguage, TempliTimeContext
from allennlp_semparse.common.action_space_walker import ActionSpaceWalker

"""
This file is meant to run standalone in the root of the project folder
This file generates logical form temlpates for TemliLanguage
"""


class LogicalFormTemplates:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.template_path = params["LF_TEMPLATE_PATH"]
        if os.path.isfile(self.template_path):
            with open(self.template_path) as f:
                self.template = json.load(f)
                # restore type int of keys
                self.template = {int(k): v for k, v in self.template.items()}
        else:
            self.template = None

    def iterate_logical_form_templates(self):
        self.params["LF_LEN"] = (
            8 if not "LF_LEN" in self.params else self.params["LF_LEN"]
        )
        self.params["MAX_LF_NUM"] = (
            5000 if not "MAX_LF_NUM" in self.params else self.params["MAX_LF_NUM"]
        )

        target_vars = set([f"${i}$" for i in range(self.params["LF_LEN"])])
        context = TempliTimeContext(
            temp_vars=target_vars, knowledge_graph={}
        )  # knowledge_graph not required to generate lf
        world = TempliLanguage(context)
        all_logical_forms = []
        walker = ActionSpaceWalker(world, params=self.params)

        for action_sequence in walker.iterate_all_action_sequences(
            max_num_logical_forms=self.params["MAX_LF_NUM"]
        ):
            yield {
                len(action_sequence): [
                    walker._world.action_sequence_to_logical_form(action_sequence)
                ]
            }

    def gen_logical_form_templates(self):
        self.params["LF_LEN"] = (
            8 if not "LF_LEN" in self.params else self.params["LF_LEN"]
        )
        self.params["MAX_LF_NUM"] = (
            5000 if not "MAX_LF_NUM" in self.params else self.params["MAX_LF_NUM"]
        )

        target_vars = set([f"${i}$" for i in range(self.params["LF_LEN"])])
        context = TempliTimeContext(
            temp_vars=target_vars, knowledge_graph={}
        )  # knowledge_graph not required to generate lf
        world = TempliLanguage(context)
        all_logical_forms = []
        walker = ActionSpaceWalker(world, max_path_length=self.params["LF_LEN"])
        action_sequences = walker.get_all_action_sequences(
            max_num_logical_forms=self.params["MAX_LF_NUM"]
        )

        # logical_form_templates: Dict[lf_len,List[lf]]
        logical_form_templates = {}
        for seq in action_sequences:
            lf_len = len(seq)
            if lf_len not in logical_form_templates:
                logical_form_templates[lf_len] = []
            logical_form_templates[lf_len] += [
                walker._world.action_sequence_to_logical_form(seq)
            ]

        return logical_form_templates

    def apply_logical_form_templates(self, min_lf_len, max_lf_len, target_vars: Set):
        max_lf_len = min(max_lf_len, max(self.template.keys()))
        target_var_str_map = {f"${idx}$": var for idx, var in enumerate(target_vars)}

        def replace_vars(lf):
            for k, v in target_var_str_map.items():
                lf = lf.replace(k, v)
            return lf if not "$" in lf else None

        def apply_lf_len(lf_len: str):
            return filter(None.__ne__, map(replace_vars, self.template[lf_len]))

        logical_forms = sum(
            [
                list(apply_lf_len(i)) if i in self.template else []
                for i in range(min_lf_len, max_lf_len + 1)
            ],
            [],
        )
        return logical_forms


if __name__ == "__main__":
    cache_params = {
        "LF_LEN": 8,
        "MAX_LF_NUM": 9999999,
        "LF_TEMPLATE_PATH": "training/cache/logical_form_templates.json",
    }

    template = LogicalFormTemplates(params=cache_params)
    logical_form_templates = template.gen_logical_form_templates()

    # Writes to file
    Path().mkdir(parents=True, exist_ok=True)
    json.dump(
        logical_form_templates,
        open(cache_params["LF_TEMPLATE_PATH"], "w"),
        indent=4,
    )