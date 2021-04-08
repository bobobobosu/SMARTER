import copy
import uuid
from datetime import datetime
from itertools import product
from typing import List, Tuple, Set

GLOBAL_START_LIM = datetime.timestamp(datetime(1900, 1, 1, 0, 0))
GLOBAL_END_LIM = datetime.timestamp(datetime(2100, 1, 1, 0, 0))
MIN_NEG = GLOBAL_START_LIM - GLOBAL_END_LIM
MAX_POS = GLOBAL_END_LIM - GLOBAL_START_LIM


class TimeVar(object):
    def __init__(self):
        self.id = uuid.uuid4()
        self.lb = GLOBAL_START_LIM
        self.ub = GLOBAL_END_LIM

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class IntervalVar(object):
    def __init__(self, name: str = None):
        self.id = uuid.uuid4()
        self.name = name if name else str(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return self.name


class Relation(object):
    def __init__(self, lhs: IntervalVar = None, rel: str = "", rhs: IntervalVar = None):
        # [lhs] = (lb, hb)
        self.id = uuid.uuid4()
        self.lhs = lhs
        self.rel = rel
        self.rhs = rhs

    def __repr__(self):
        return f"{str(self.lhs)[:10]} {self.rel} {str(self.rhs)[:10]}"


class Constraint(object):
    def __init__(self):
        # [lhs] = (lb, hb)
        self.id = uuid.uuid4()
        self.lhs = []  # type: List[Tuple[TimeVar,int]]
        self.rhs = (MIN_NEG, MAX_POS)  # type: Tuple(int,int)


class TimeInterval(object):
    def __init__(self, other: "TimeInterval" = None, name: str = None) -> None:
        # This constructor creates a new interval based on the constraints of other
        # make sure that newer constraints/relations (that depends on the proir ones) are at the end of the list
        # 2 things saved: list of constraints and list of allen_relations

        # Initialize start end variables
        self.startvar = TimeVar()
        self.endvar = TimeVar()
        self.intervalvar = IntervalVar(name=name)

        # Initialize constraints
        self.constraints_list = (
            copy.deepcopy(other.constraints_list) if other else [[]]
        )  # type: List[List[Constraint]]
        for constraints in self.constraints_list:
            constraints += self.gen_intrinstic_constraints()

        # Initialize allen_relations (or of ands -> DNF format)
        self.allen_relations_list = (
            copy.deepcopy(other.allen_relations_list) if other else [[]]
        )  # List[List[Relation]]

    def gen_intrinstic_constraints(self) -> List[Constraint]:
        # Constraint: end - start >= 0
        intr_constraint = Constraint()
        intr_constraint.lhs += [(self.endvar, 1), (self.startvar, -1)]
        intr_constraint.rhs = (0, MAX_POS)
        return [intr_constraint]

    def replace_intervalid(self, targetvar: IntervalVar, withvar: IntervalVar):
        # replace allen_relations_list
        for relations in self.allen_relations_list:
            for relation in relations:
                relation.lhs = withvar if relation.lhs == targetvar else relation.lhs
                relation.rhs = withvar if relation.rhs == targetvar else relation.rhs

    def replace_id(self, targetvar: TimeVar, withvar: TimeVar):

        # replace start, end variables
        self.startvar = withvar if self.startvar == targetvar else self.startvar
        self.endvar = withvar if self.endvar == targetvar else self.endvar
        self.startvar = withvar if self.startvar == targetvar else self.startvar
        self.endvar = withvar if self.endvar == targetvar else self.endvar

        # replace constraints
        for constraints in self.constraints_list:
            for constraint in constraints:
                constraint.lhs = [
                    (withvar, i[1]) if i[0] == targetvar else i for i in constraint.lhs
                ]

    def intersect(self, other: "TimeInterval") -> None:
        # current startVar and endVar don't change
        product_constraints = product(self.constraints_list, other.constraints_list)
        self.constraints_list = [sum(i, []) for i in product_constraints]
        self.replace_id(other.startvar, self.startvar)
        self.replace_id(other.endvar, self.endvar)

        # update allen_relations
        product_relations = product(self.allen_relations_list, other.allen_relations_list)
        self.allen_relations_list = [sum(i, []) for i in product_relations]
        self.replace_intervalid(other.intervalvar, self.intervalvar)

    def union(self, other: "TimeInterval") -> None:
        # current startVar and endVar don't change
        self.constraints_list += other.constraints_list
        self.replace_id(other.startvar, self.startvar)
        self.replace_id(other.endvar, self.endvar)

        # update allen_relations
        self.allen_relations_list += other.allen_relations_list
        self.replace_intervalid(other.intervalvar, self.intervalvar)

    def append_constraint(self, constraint: Constraint) -> None:
        for constraints in self.constraints_list:
            constraints.append(constraint)

    def get_interval_variables(self) -> List[Set[IntervalVar]]:
        vars = {self.intervalvar} # TODO not sure if this is a bug but this works
        for relations in self.allen_relations_list:
            for relation in relations:
                vars |= {relation.lhs, relation.rhs}
        return vars

    def get_variables(self) -> List[Set[TimeVar]]:
        vars = set()
        for constraints in self.constraints_list:
            for constraint in constraints:
                vars |= set([i[0] for i in constraint.lhs])
        return vars

    def set_start(self, startdatetime: int) -> None:
        self.startvar.lb = self.startvar.ub = startdatetime

    def set_end(self, enddatetime: int) -> None:
        self.endvar.lb = self.endvar.ub = enddatetime

    def __hash__(self):
        return hash((self.startvar, self.endvar, self.id))

    def __eq__(self, other):
        return (self.startvar, self.endvar, self.id) == (other.startvar, other.endvar, other.id)

### LINES GENERATED BY templi_primitives_types.txt

class InterTimeInterval(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_D(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_F(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_M(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_O(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_P(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_S(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_d(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_e(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_f(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_m(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_o(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_p(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval
class TimeInterval_s(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval

### LINES GENERATED BY templi_primitives_types.txt
