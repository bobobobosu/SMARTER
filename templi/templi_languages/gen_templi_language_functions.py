import pprint
"""
This file is meant to run standalone in this folder
This file generates all predicate for TemliLanguage
"""
# The naming used here is from https://www.ics.uci.edu/~alspaugh/cls/shr/allen.html
uci_rels = ["p", "m", "o", "F", "D", "s", "e", "S", "d", "f", "O", "M", "P"]
composition_table = [
        ["p", "p", "p", "p", "p", "p", "p", "p", "pmosd", "pmosd", "pmosd", "pmosd", "pmoFDseSdfOMP"],
        ["p", "p", "p", "p", "p", "m", "m", "m", "osd", "osd", "osd", "Fef", "DSOMP"],
        ["p", "p", "pmo", "pmo", "pmoFD", "o", "o", "oFD", "osd", "osd", "oFDseSdfO", "DSO", "DSOMP"],
        ["p", "m", "o", "F", "D", "o", "F", "D", "osd", "Fef", "DSO", "DSO", "DSOMP"],
        ["pmoFD", "oFD", "oFD", "D", "D", "oFD", "D", "D", "oFDseSdfO", "DSO", "DSO", "DSO", "DSOMP"],
        ["p", "p", "pmo", "pmo", "pmoFD", "s", "s", "seS", "d", "d", "dfO", "M", "P"],
        ["p", "m", "o", "F", "D", "s", "e", "S", "d", "f", "O", "M", "P"],
        ["pmoFD", "oFD", "oFD", "D", "D", "seS", "S", "S", "dfO", "O", "O", "M", "P"],
        ["p", "p", "pmosd", "pmosd", "pmoFDseSdfOMP", "d", "d", "dfOMP", "d", "d", "dfOMP", "P", "P"],
        ["p", "m", "osd", "Fef", "DSOMP", "d", "f", "OMP", "d", "f", "OMP", "P", "P"],
        ["pmoFD", "oFD", "oFDseSdfO", "DSO", "DSOMP", "dfO", "O", "OMP", "dfO", "O", "OMP", "P", "P"],
        ["pmoFD", "seS", "dfO", "M", "P", "dfO", "M", "P", "dfO", "M", "P", "P", "P"],
        ["pmoFDseSdfOMP", "dfOMP", "dfOMP", "P", "P", "dfOMP", "P", "P", "dfOMP", "P", "P", "P", "P"],
    ]
op_to_func = {
    "p": "precedes",
    "m": "meets",
    "o": "overlaps",
    "F": "finished_by",
    "D": "contains",
    "s": "starts",
    "e": "equals",
    "S": "started_by",
    "d": "during",
    "f": "finishes",
    "O": "overlapped_by",
    "M": "met_by",
    "P": "preceded_by",
}

func_lines = []
pred_lines = []
type_lines = []
valid_com = {}
for r, row in enumerate(composition_table):
    valid_com[uci_rels[r]] = ""
    for c, col in enumerate(row):
        lhs = uci_rels[r]
        rhs = uci_rels[c]
        if lhs == 'e' or rhs == 'e': # equals function has no function
            continue
        cop = col
        if lhs != cop:
            valid_com[uci_rels[r]] += rhs
            func_line = f"""
    def func_{lhs}_{rhs}(self, lhs: TimeInterval_{rhs}) -> TimeInterval_{lhs}:
        return self.{op_to_func[lhs]}(lhs)"""
            pred_line = f"""
        self.add_predicate("func_{lhs}_{rhs}", self.func_{lhs}_{rhs})"""
            type_line = f"""
class TimeInterval_{lhs}(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval"""
            pred_lines += [pred_line]
            func_lines += [func_line]
            type_lines += [type_line]

            # Support for start
            func_line = f"""
    def start_{rhs}(self, lhs: TimeInterval_{rhs}) -> TimeInterval:
        return self.{op_to_func[lhs]}(lhs)"""   
            pred_line = f"""
        self.add_predicate("start_{rhs}", self.start_{rhs})"""  
            pred_lines += [pred_line]
            func_lines += [func_line]     

            # Support for terminal
            func_line = f"""
    def term_{lhs}(self, lhs: InterTimeInterval) -> TimeInterval_{lhs}:
        return self.{op_to_func[lhs]}(lhs)"""   
            pred_line = f"""
        self.add_predicate("term_{lhs}", self.term_{lhs})"""  
            pred_lines += [pred_line]
            func_lines += [func_line]

## Intersection
pred_line = f"""
        self.add_predicate("func_intersection", self.func_intersection)"""  
func_line = f"""
    def func_intersection(self, lhs: TimeInterval, rhs: TimeInterval) -> InterTimeInterval:
        return self.intersection(lhs, rhs)"""
pred_lines += [pred_line]
func_lines += [func_line]

## Union
pred_line = f"""
        self.add_predicate("func_union", self.func_union)"""  
func_line = f"""
    def func_union(self, lhs: TimeInterval, rhs: TimeInterval) -> InterTimeInterval:
        return self.union(lhs, rhs)"""
pred_lines += [pred_line]
func_lines += [func_line]  

## Reset to TimeInterval
pred_line = f"""
        self.add_predicate("reset", self.reset)"""  
func_line = f"""
    def reset(self, lhs: InterTimeInterval) -> TimeInterval:
        return lhs"""
pred_lines += [pred_line]
func_lines += [func_line]  

## Intermediate type
type_line = f"""
class InterTimeInterval(object):
    def __init__(self, time_interval: "TimeInterval") -> None:
        self.time_interval = time_interval"""
type_lines += [type_line]     

with open("templi/templi_languages/templi_language_predicates.txt", "w") as text_file:
    print(''.join(sorted(list(set(pred_lines)))), file=text_file)
with open("templi/templi_languages/templi_language_functions.txt", "w") as text_file:
    print(''.join(sorted(list(set(func_lines)))), file=text_file)
with open("templi/templi_languages/templi_primitives_types.txt", "w") as text_file:
    print(''.join(sorted(list(set(type_lines)))), file=text_file)

pprint.pprint(valid_com)