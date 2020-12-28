from typing import List, Callable, Dict, Tuple, Set, Type
from templi.templi_languages.templi_time_context import TempliTimeContext
from allennlp_semparse import DomainLanguage
from datetime import datetime, timedelta
from allennlp_semparse.common.util import lisp_to_nested_expression
from allennlp_semparse.common import Date, ExecutionError, MONTH_NUMBERS
import copy
import uuid
from uuid import UUID
from itertools import product
from templi.templi_languages.allen_algebra import infer_relation
from templi.templi_languages.templi_primitives import TimeInterval, FinalTimeInterval, Constraint, Relation, MIN_NEG, MAX_POS

"""
TemporalContext: contains event/time mentioned in the sentence and 
commonly used time expression hierarchy e.g. 2020, january, weekdays, centuries...
"""


class TempliLanguage(DomainLanguage):
    def __init__(self, time_context: TempliTimeContext) -> None:
        super().__init__(start_types={TimeInterval})

        # Terminal Function
        self.add_predicate("terminate", self.terminate)  # terminate($Interval)->$FinalInterval

        # TODO Superlative Functions
        # self.add_predicate(
        #     "offset_cnt", self.offset_cnt
        # )  # offset_cnt($Interval->$Bool, $Int)->$Interval
        # self.add_predicate("max", self.max)  # max($Interval->$Bool)->$Interval
        # self.add_predicate("min", self.min)  # min($Interval->$Bool)->$Interval

        # TODO Functions on booleans
        # self.add_predicate("complement", self.complement)  # complement($Bool)->$Bool
        self.add_predicate("intersection", self.intersection)  # intersection($Bool,$Bool)->$Bool
        self.add_predicate("union", self.union)  # union($Bool,$Bool)->$Bool

        # Functions on operators
        # self.add_predicate("converse", self.converse)  # (converse($Operator))->$Operator

        # Functions on intervals
        # we use uci naming here
        # source: https://www.ics.uci.edu/~alspaugh/cls/shr/allen.html
        # self.add_predicate("offset", self.offset)  # offset($Interval, $Duration)->$Bool
        self.add_predicate("e", self.equals)
        self.add_predicate("p", self.precedes)
        self.add_predicate("P", self.preceded_by)
        self.add_predicate("m", self.meets)
        self.add_predicate("M", self.met_by)
        self.add_predicate("f", self.finishes)
        self.add_predicate("F", self.finished_by)
        self.add_predicate("d", self.during)
        self.add_predicate("D", self.contains)
        self.add_predicate("s", self.starts)
        self.add_predicate("S", self.started_by)
        self.add_predicate("o", self.overlaps)
        self.add_predicate("O", self.overlapped_by)


        # constant functions for variables
        self.constant_vars = {}
        for var in time_context.temp_vars:
            self.constant_vars[var] = TimeInterval(name=var)
            self.add_constant(var, self.constant_vars[var], type_=TimeInterval)
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
                rel = infer_relation(denotation, self.constant_vars[target_var].intervalvar)
                if ''.join(rel) != target_rel:
                    return False
        except ExecutionError as error:
            print(f"Failed to execute: {logical_form}. Error: {error}")
            return False
        return True

    def terminate(self, lhs: TimeInterval) -> FinalTimeInterval:
        return FinalTimeInterval(lhs)

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

# TODO logical_form_to_intervaltree
# def logical_form_to_intervaltree(s: str) -> List[str]:
#     nested = lisp_to_nested_expression(s)
#
#     def filter_interval(tree: intervaltree) -> intervaltree:
