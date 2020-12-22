# THIS FILE IS ONLY FOR TESTING

def logical_form_to_constrints(s: str) -> List[str]:
    nested = lisp_to_nested_expression(s)
    id_cnter = 0

    def genid():
        nonlocal id_cnter
        id_cnter += 1
        return f"{id_cnter}_s", f"{id_cnter}_e"

    def collect_constraints(nested_exp) -> Tuple[int, List[str]]:
        startid, endid = genid()
        constraints = []

        # Superlative Functions
        if nested_exp[0] == "offset_cnt":
            pass
        elif nested_exp[0] == "max":
            pass
        elif nested_exp[0] == "min":
            pass
        # Functions on booleans
        elif nested_exp[0] == "complement":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])

        elif nested_exp[0] == "intersection":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            rsid, reid, rconstraints = collect_constraints(nested_exp[2])
            constraints += [i.replace(lsid, startid).replace(leid, endid) for i in lconstraints]
            constraints += [i.replace(rsid, startid).replace(reid, endid) for i in rconstraints]

        elif nested_exp[0] == "union":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            rsid, reid, rconstraints = collect_constraints(nested_exp[2])

        # Functions on intervals
        elif nested_exp[0] == "offset":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [
                f"{startid}=={lsid}+{nested_exp[2]}",
                f"{endid}=={leid}+{nested_exp[2]}",
            ]
            constraints += lconstraints
        elif nested_exp[0] == "equals":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{startid}=={lsid}", f"{endid}=={leid}"]
            constraints += lconstraints
        elif nested_exp[0] == "precedes":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{endid}<{lsid}"]
            constraints += lconstraints
        elif nested_exp[0] == "preceded_by":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{leid}<{startid}"]
            constraints += lconstraints
        elif nested_exp[0] == "meets":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{endid}=={lsid}"]
            constraints += lconstraints
        elif nested_exp[0] == "met_by":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{leid}=={startid}"]
            constraints += lconstraints
        elif nested_exp[0] == "finishes":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{endid}=={leid}", f"{startid}>{lsid}"]
            constraints += lconstraints
        elif nested_exp[0] == "finished_by":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{leid}=={endid}", f"{lsid}>{startid}"]
            constraints += lconstraints
        elif nested_exp[0] == "during":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{startid}>{lsid}", f"{endid}<{leid}"]
            constraints += lconstraints
        elif nested_exp[0] == "contains":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{startid}<{lsid}", f"{endid}>{leid}"]
            constraints += lconstraints
        elif nested_exp[0] == "starts":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{startid}=={lsid}", f"{endid}<{leid}"]
            constraints += lconstraints
        elif nested_exp[0] == "started_by":
            lsid, leid, lconstraints = collect_constraints(nested_exp[1])
            constraints += [f"{lsid}=={startid}", f"{leid}<{endid}"]
            constraints += lconstraints

        # Unbounded variables
        else:
            constraints += [f"{startid}<={endid}"]

        return startid, endid, constraints

    _, _, constraints = collect_constraints(nested)
    return constraints


# fff = lisp_to_nested_expression("(intersection (during (clothe_dry)) (contains (killed)))")
# # constraintd = logical_form_to_constrints("(intersection (during (clothes_dry)) (contains (killed)))")
# # print(constraintd)
# constraintd = logical_form_to_constrints(
#     "(intersection (during (clothes_dry)) (contains (during (intersection (during Thursday) (during evening)))))"
# )
# print(constraintd)
