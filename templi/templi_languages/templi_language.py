from typing import List, Callable, Dict, Tuple, Set, Type
from templi.templi_languages.templi_time_context import TempliTimeContext
from allennlp_semparse.domain_languages.domain_language import (
    DomainLanguage,
    PredicateType,
    predicate,
)
from datetime import datetime, timedelta
from allennlp_semparse.common.util import lisp_to_nested_expression
from allennlp_semparse.common import Date, ExecutionError, MONTH_NUMBERS
import copy
import uuid
from uuid import UUID
from itertools import product
from templi.templi_languages.allen_algebra import infer_relation
from templi.templi_languages.templi_primitives import *

"""
TemporalContext: contains event/time mentioned in the sentence and 
commonly used time expression hierarchy e.g. 2020, january, weekdays, centuries...
"""


class TempliLanguage(DomainLanguage):
    def __init__(self, time_context: TempliTimeContext) -> None:
        super().__init__(
            start_types={
                TimeInterval,
                InterTimeInterval
            }
        )

        # TODO Superlative Functions
        # self.add_predicate(
        #     "offset_cnt", self.offset_cnt
        # )  # offset_cnt($Interval->$Bool, $Int)->$Interval
        # self.add_predicate("max", self.max)  # max($Interval->$Bool)->$Interval
        # self.add_predicate("min", self.min)  # min($Interval->$Bool)->$Interval

        # TODO Functions on booleans
        # self.add_predicate("complement", self.complement)  # complement($Bool)->$Bool
        # union and intersection are declared in the generated block

        # Functions on operators
        # defined below

        # Functions on intervals
        # we use uci naming here
        # source: https://www.ics.uci.edu/~alspaugh/cls/shr/allen.html
        # defined below

        self.table_graph = time_context.get_table_knowledge_graph()

        # constant functions for variables
        self.constant_vars = {}
        for var in time_context.temp_vars:
            self.constant_vars[var] = TimeInterval(name=var)
            self.add_constant(
                f"const_{var}", self.constant_vars[var], type_=TimeInterval
            )  # converted to TimeInterval by reset
        pass

    def evaluate_logical_form(
        self, logical_form: str, target_rels: Dict[str, str]
    ) -> bool:
        """
        Takes a logical form, and the dict {some_event:relation}, and returns True iff the logical form
        executes to the target dict
        """
        try:
            denotation = self.execute(logical_form)
            for target_var, target_rel in target_rels.items():
                rel = infer_relation(
                    denotation, self.constant_vars[target_var].intervalvar
                )
                if "".join(rel) != target_rel:
                    return False
        except ExecutionError as error:
            print(f"Failed to execute: {logical_form}. Error: {error}")
            return False
        return True

    def evaluate_logical_form_partial_match(
        self, logical_form: str, target_rels: Dict[str, str]
    ) -> int:
        """
        Takes a logical form, and the dict {some_event:relation}, and returns True iff the logical form
        executes to the target dict
        """
        match = 0
        try:
            denotation = self.execute(logical_form)
            for target_var, target_rel in target_rels.items():
                rel = infer_relation(
                    denotation, self.constant_vars[target_var].intervalvar
                )
                if "".join(rel) == target_rel:
                    match += 1
        except ExecutionError as error:
            print(f"Failed to execute: {logical_form}. Error: {error}")
            return match
        return match

    def evaluate_logical_form_holistically(
        self,
        denotation_dict: Dict[str, TimeInterval],
        target_rels: Dict[str, Dict[str, str]],
    ):
        # first we have to bind variables of events among denotations so that the merged graph would be connected
        # we do nothing here because when dataset reader takes the input, the TimeIntervals are guaranteed to be
        # binded accross instances within one data TODO there may be a better way to do this

        # second we merge the allen_relations_list and constraints_list of each TimeInterval
        merged_allen_relations_list = [
            sum(j, [])
            for j in product([i.allen_relations_list for i in denotation_dict.values()])
        ]
        merged_constraints_list = [
            sum(j, [])
            for j in product([i.constraints_list for i in denotation_dict.values()])
        ]

        # third we infer the relations between each pair of events
        holistic_denotation = {}
        full_hits = sum([len(v) for k, v in target_rels.items()])
        hits = 0
        for main_var, v in target_rels.items():
            holistic_denotation[main_var] = {}
            for target_var, rel in v.items():
                for allen_relations_list in merged_allen_relations_list:
                    denotation_dict[
                        main_var
                    ].allen_relations_list = allen_relations_list
                    # we have to get the relations each pair of events correct
                    rel = infer_relation(
                        denotation_dict[main_var],
                        self.constant_vars[target_var].intervalvar,
                    )
                    holistic_denotation[main_var][target_var] = rel
                    if "".join(rel) == target_rels[main_var][target_var]:
                        hits += 1

        return hits / full_hits if full_hits > 0 else 1, holistic_denotation

    def union(self, lhs: TimeInterval, rhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval()
        hs.union(lhs)
        hs.union(rhs)
        return hs

    def intersection(self, lhs: TimeInterval, rhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval()
        hs.intersect(lhs)
        hs.intersect(rhs)
        return hs

    def equals(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start - lhs_start = 0
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (0, 0)
        hs.append_constraint(constraint)
        # hs_end - lhs_end = 0
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (0, 0)
        hs.append_constraint(constraint)

        # this equals lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "e", lhs.intervalvar))
        return hs

    def precedes(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_end - lhs_start <= 0
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)

        # this precedes lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "p", lhs.intervalvar))

        return hs

    def preceded_by(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start - lhs_end >= 0
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (0, MAX_POS)
        hs.append_constraint(constraint)

        # this preceded_by lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "P", lhs.intervalvar))

        return hs

    def meets(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_end == lhs_start
        hs.replace_id(hs.endvar, lhs.startvar)

        # this meets lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "m", lhs.intervalvar))

        return hs

    def met_by(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start == lhs_end
        hs.replace_id(hs.startvar, lhs.endvar)

        # this met_by lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "M", lhs.intervalvar))

        return hs

    def finishes(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start - lhs_start >=0
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (0, MAX_POS)
        hs.append_constraint(constraint)
        # hs_end == lhs_end
        hs.replace_id(hs.endvar, lhs.endvar)

        # this finishes lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "f", lhs.intervalvar))

        return hs

    def finished_by(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start - lhs_start <= 0
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # hs_end == lhs_end
        hs.replace_id(hs.endvar, lhs.endvar)

        # this finished_by lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "F", lhs.intervalvar))

        return hs

    def during(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start - lhs_start >= 0
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (0, MAX_POS)
        hs.append_constraint(constraint)
        # hs_end - lhs_end <= 0
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)

        # this during lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "d", lhs.intervalvar))

        return hs

    def contains(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start - lhs_start <= 0
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # hs_end - lhs_end >= 0
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (0, MAX_POS)
        hs.append_constraint(constraint)

        # this contains lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "D", lhs.intervalvar))

        return hs

    def starts(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_end - lhs_end <=0
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # hs_start == lhs_start
        hs.replace_id(hs.startvar, lhs.startvar)

        # this starts lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "s", lhs.intervalvar))

        return hs

    def started_by(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_end - lhs_end >=0
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (0, MAX_POS)
        hs.append_constraint(constraint)
        # hs_start == lhs_start
        hs.replace_id(hs.startvar, lhs.startvar)

        # this started_by lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "S", lhs.intervalvar))

        return hs

    def overlaps(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # hs_start < lhs_start
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.startvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # lhs_start < hs_end
        constraint = Constraint()
        constraint.lhs += [(lhs.startvar, 1), (hs.endvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # hs_end < lhs_end
        constraint = Constraint()
        constraint.lhs += [(hs.endvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)

        # this overlaps lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "o", lhs.intervalvar))

        return hs

    def overlapped_by(self, lhs: TimeInterval) -> TimeInterval:
        hs = TimeInterval(lhs)
        # lhs_start < hs_start
        constraint = Constraint()
        constraint.lhs += [(lhs.startvar, 1), (hs.startvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # hs_start < lhs_end
        constraint = Constraint()
        constraint.lhs += [(hs.startvar, 1), (lhs.endvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)
        # lhs_end < hs_end
        constraint = Constraint()
        constraint.lhs += [(lhs.endvar, 1), (hs.endvar, -1)]
        constraint.rhs = (MIN_NEG, 0)
        hs.append_constraint(constraint)

        # this overlapped_by lhs
        for relations in hs.allen_relations_list:
            relations.append(Relation(hs.intervalvar, "O", lhs.intervalvar))

        return hs

    ### LINES GENERATED BY templi_language_functions.txt

            
    @predicate
    def func_D_M(self, lhs: TimeInterval_M) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_O(self, lhs: TimeInterval_O) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_P(self, lhs: TimeInterval_P) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_d(self, lhs: TimeInterval_d) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_f(self, lhs: TimeInterval_f) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_m(self, lhs: TimeInterval_m) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_o(self, lhs: TimeInterval_o) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_p(self, lhs: TimeInterval_p) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_D_s(self, lhs: TimeInterval_s) -> TimeInterval_D:
        return self.contains(lhs)
            
    @predicate
    def func_F_D(self, lhs: TimeInterval_D) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_M(self, lhs: TimeInterval_M) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_O(self, lhs: TimeInterval_O) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_P(self, lhs: TimeInterval_P) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_S(self, lhs: TimeInterval_S) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_d(self, lhs: TimeInterval_d) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_f(self, lhs: TimeInterval_f) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_m(self, lhs: TimeInterval_m) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_o(self, lhs: TimeInterval_o) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_p(self, lhs: TimeInterval_p) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_F_s(self, lhs: TimeInterval_s) -> TimeInterval_F:
        return self.finished_by(lhs)
            
    @predicate
    def func_M_D(self, lhs: TimeInterval_D) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_M(self, lhs: TimeInterval_M) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_O(self, lhs: TimeInterval_O) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_P(self, lhs: TimeInterval_P) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_S(self, lhs: TimeInterval_S) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_d(self, lhs: TimeInterval_d) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_m(self, lhs: TimeInterval_m) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_o(self, lhs: TimeInterval_o) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_p(self, lhs: TimeInterval_p) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_M_s(self, lhs: TimeInterval_s) -> TimeInterval_M:
        return self.met_by(lhs)
            
    @predicate
    def func_O_D(self, lhs: TimeInterval_D) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_F(self, lhs: TimeInterval_F) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_M(self, lhs: TimeInterval_M) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_O(self, lhs: TimeInterval_O) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_P(self, lhs: TimeInterval_P) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_S(self, lhs: TimeInterval_S) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_d(self, lhs: TimeInterval_d) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_m(self, lhs: TimeInterval_m) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_o(self, lhs: TimeInterval_o) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_p(self, lhs: TimeInterval_p) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_O_s(self, lhs: TimeInterval_s) -> TimeInterval_O:
        return self.overlapped_by(lhs)
            
    @predicate
    def func_P_d(self, lhs: TimeInterval_d) -> TimeInterval_P:
        return self.preceded_by(lhs)
            
    @predicate
    def func_P_m(self, lhs: TimeInterval_m) -> TimeInterval_P:
        return self.preceded_by(lhs)
            
    @predicate
    def func_P_o(self, lhs: TimeInterval_o) -> TimeInterval_P:
        return self.preceded_by(lhs)
            
    @predicate
    def func_P_p(self, lhs: TimeInterval_p) -> TimeInterval_P:
        return self.preceded_by(lhs)
            
    @predicate
    def func_P_s(self, lhs: TimeInterval_s) -> TimeInterval_P:
        return self.preceded_by(lhs)
            
    @predicate
    def func_S_D(self, lhs: TimeInterval_D) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_F(self, lhs: TimeInterval_F) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_M(self, lhs: TimeInterval_M) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_O(self, lhs: TimeInterval_O) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_P(self, lhs: TimeInterval_P) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_d(self, lhs: TimeInterval_d) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_f(self, lhs: TimeInterval_f) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_m(self, lhs: TimeInterval_m) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_o(self, lhs: TimeInterval_o) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_p(self, lhs: TimeInterval_p) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_S_s(self, lhs: TimeInterval_s) -> TimeInterval_S:
        return self.started_by(lhs)
            
    @predicate
    def func_d_D(self, lhs: TimeInterval_D) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_F(self, lhs: TimeInterval_F) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_M(self, lhs: TimeInterval_M) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_O(self, lhs: TimeInterval_O) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_P(self, lhs: TimeInterval_P) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_S(self, lhs: TimeInterval_S) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_m(self, lhs: TimeInterval_m) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_o(self, lhs: TimeInterval_o) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_d_p(self, lhs: TimeInterval_p) -> TimeInterval_d:
        return self.during(lhs)
            
    @predicate
    def func_e_D(self, lhs: TimeInterval_D) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_F(self, lhs: TimeInterval_F) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_M(self, lhs: TimeInterval_M) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_O(self, lhs: TimeInterval_O) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_P(self, lhs: TimeInterval_P) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_S(self, lhs: TimeInterval_S) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_d(self, lhs: TimeInterval_d) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_f(self, lhs: TimeInterval_f) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_m(self, lhs: TimeInterval_m) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_o(self, lhs: TimeInterval_o) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_p(self, lhs: TimeInterval_p) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_e_s(self, lhs: TimeInterval_s) -> TimeInterval_e:
        return self.equals(lhs)
            
    @predicate
    def func_f_D(self, lhs: TimeInterval_D) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_F(self, lhs: TimeInterval_F) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_M(self, lhs: TimeInterval_M) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_O(self, lhs: TimeInterval_O) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_P(self, lhs: TimeInterval_P) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_S(self, lhs: TimeInterval_S) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_d(self, lhs: TimeInterval_d) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_m(self, lhs: TimeInterval_m) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_o(self, lhs: TimeInterval_o) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_p(self, lhs: TimeInterval_p) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_f_s(self, lhs: TimeInterval_s) -> TimeInterval_f:
        return self.finishes(lhs)
            
    @predicate
    def func_m_D(self, lhs: TimeInterval_D) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_F(self, lhs: TimeInterval_F) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_M(self, lhs: TimeInterval_M) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_O(self, lhs: TimeInterval_O) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_P(self, lhs: TimeInterval_P) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_d(self, lhs: TimeInterval_d) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_f(self, lhs: TimeInterval_f) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_m(self, lhs: TimeInterval_m) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_o(self, lhs: TimeInterval_o) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_m_p(self, lhs: TimeInterval_p) -> TimeInterval_m:
        return self.meets(lhs)
            
    @predicate
    def func_o_D(self, lhs: TimeInterval_D) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_F(self, lhs: TimeInterval_F) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_M(self, lhs: TimeInterval_M) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_O(self, lhs: TimeInterval_O) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_P(self, lhs: TimeInterval_P) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_S(self, lhs: TimeInterval_S) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_d(self, lhs: TimeInterval_d) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_f(self, lhs: TimeInterval_f) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_m(self, lhs: TimeInterval_m) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_o(self, lhs: TimeInterval_o) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_o_p(self, lhs: TimeInterval_p) -> TimeInterval_o:
        return self.overlaps(lhs)
            
    @predicate
    def func_p_M(self, lhs: TimeInterval_M) -> TimeInterval_p:
        return self.precedes(lhs)
            
    @predicate
    def func_p_O(self, lhs: TimeInterval_O) -> TimeInterval_p:
        return self.precedes(lhs)
            
    @predicate
    def func_p_P(self, lhs: TimeInterval_P) -> TimeInterval_p:
        return self.precedes(lhs)
            
    @predicate
    def func_p_d(self, lhs: TimeInterval_d) -> TimeInterval_p:
        return self.precedes(lhs)
            
    @predicate
    def func_p_f(self, lhs: TimeInterval_f) -> TimeInterval_p:
        return self.precedes(lhs)
            
    @predicate
    def func_s_D(self, lhs: TimeInterval_D) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_F(self, lhs: TimeInterval_F) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_M(self, lhs: TimeInterval_M) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_O(self, lhs: TimeInterval_O) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_P(self, lhs: TimeInterval_P) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_S(self, lhs: TimeInterval_S) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_d(self, lhs: TimeInterval_d) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_f(self, lhs: TimeInterval_f) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_m(self, lhs: TimeInterval_m) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_o(self, lhs: TimeInterval_o) -> TimeInterval_s:
        return self.starts(lhs)
            
    @predicate
    def func_s_p(self, lhs: TimeInterval_p) -> TimeInterval_s:
        return self.starts(lhs)
    @predicate
    def func_intersection_2(self , arg0: InterTimeInterval, arg1: InterTimeInterval) -> TimeInterval:
        return self.intersection(arg0, arg1)
    @predicate
    def func_intersection_3(self , arg0: InterTimeInterval, arg1: InterTimeInterval, arg2: InterTimeInterval) -> TimeInterval:
        return self.intersection(arg2, self.intersection(arg0, arg1))
    @predicate
    def func_union_2(self , arg0: InterTimeInterval, arg1: InterTimeInterval) -> TimeInterval:
        return self.union(arg0, arg1)
    @predicate
    def func_union_3(self , arg0: InterTimeInterval, arg1: InterTimeInterval, arg2: InterTimeInterval) -> TimeInterval:
        return self.union(arg2, self.union(arg0, arg1))
    @predicate
    def op1_D(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op1_F(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op1_M(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op1_O(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op1_P(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.preceded_by(lhs)
    @predicate
    def op1_S(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op1_d(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op1_e(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op1_f(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op1_m(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op1_o(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op1_p(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.precedes(lhs)
    @predicate
    def op1_s(self, lhs: TimeInterval) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_D_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_D_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.contains(lhs)
    @predicate
    def op2_end_F_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_F_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.finished_by(lhs)
    @predicate
    def op2_end_M_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_M_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.met_by(lhs)
    @predicate
    def op2_end_O_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_O_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.overlapped_by(lhs)
    @predicate
    def op2_end_P_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.preceded_by(lhs)
    @predicate
    def op2_end_P_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.preceded_by(lhs)
    @predicate
    def op2_end_P_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.preceded_by(lhs)
    @predicate
    def op2_end_P_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.preceded_by(lhs)
    @predicate
    def op2_end_P_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.preceded_by(lhs)
    @predicate
    def op2_end_S_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_S_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.started_by(lhs)
    @predicate
    def op2_end_d_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_d_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.during(lhs)
    @predicate
    def op2_end_e_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_e_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.equals(lhs)
    @predicate
    def op2_end_f_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_f_s(self, lhs: TimeInterval_s) -> InterTimeInterval:
        return self.finishes(lhs)
    @predicate
    def op2_end_m_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_m_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.meets(lhs)
    @predicate
    def op2_end_o_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_o_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.overlaps(lhs)
    @predicate
    def op2_end_p_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.precedes(lhs)
    @predicate
    def op2_end_p_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.precedes(lhs)
    @predicate
    def op2_end_p_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.precedes(lhs)
    @predicate
    def op2_end_p_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.precedes(lhs)
    @predicate
    def op2_end_p_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.precedes(lhs)
    @predicate
    def op2_end_s_D(self, lhs: TimeInterval_D) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_F(self, lhs: TimeInterval_F) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_M(self, lhs: TimeInterval_M) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_O(self, lhs: TimeInterval_O) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_P(self, lhs: TimeInterval_P) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_S(self, lhs: TimeInterval_S) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_d(self, lhs: TimeInterval_d) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_f(self, lhs: TimeInterval_f) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_m(self, lhs: TimeInterval_m) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_o(self, lhs: TimeInterval_o) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_end_s_p(self, lhs: TimeInterval_p) -> InterTimeInterval:
        return self.starts(lhs)
    @predicate
    def op2_start_D(self, lhs: TimeInterval) -> TimeInterval_D:
        return self.contains(lhs)
    @predicate
    def op2_start_F(self, lhs: TimeInterval) -> TimeInterval_F:
        return self.finished_by(lhs)
    @predicate
    def op2_start_M(self, lhs: TimeInterval) -> TimeInterval_M:
        return self.met_by(lhs)
    @predicate
    def op2_start_O(self, lhs: TimeInterval) -> TimeInterval_O:
        return self.overlapped_by(lhs)
    @predicate
    def op2_start_P(self, lhs: TimeInterval) -> TimeInterval_P:
        return self.preceded_by(lhs)
    @predicate
    def op2_start_S(self, lhs: TimeInterval) -> TimeInterval_S:
        return self.started_by(lhs)
    @predicate
    def op2_start_d(self, lhs: TimeInterval) -> TimeInterval_d:
        return self.during(lhs)
    @predicate
    def op2_start_e(self, lhs: TimeInterval) -> TimeInterval_e:
        return self.equals(lhs)
    @predicate
    def op2_start_f(self, lhs: TimeInterval) -> TimeInterval_f:
        return self.finishes(lhs)
    @predicate
    def op2_start_m(self, lhs: TimeInterval) -> TimeInterval_m:
        return self.meets(lhs)
    @predicate
    def op2_start_o(self, lhs: TimeInterval) -> TimeInterval_o:
        return self.overlaps(lhs)
    @predicate
    def op2_start_p(self, lhs: TimeInterval) -> TimeInterval_p:
        return self.precedes(lhs)
    @predicate
    def op2_start_s(self, lhs: TimeInterval) -> TimeInterval_s:
        return self.starts(lhs)

    ### LINES GENERATED BY templi_language_functions.txt


# TODO logical_form_to_intervaltree
# def logical_form_to_intervaltree(s: str) -> List[str]:
#     nested = lisp_to_nested_expression(s)
#
#     def filter_interval(tree: intervaltree) -> intervaltree:
