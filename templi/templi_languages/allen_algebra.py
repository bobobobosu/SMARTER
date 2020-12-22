from typing import Optional, Set
from templi.templi_languages.templi_primitives import TimeInterval, IntervalVar
from networkx.algorithms.simple_paths import all_simple_paths
import networkx as nx
import itertools

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
converse_table = {'p':'P','m':'M','o':'O','F':'f','D':'d','s':'S'}
timeml_to_uci_dict = {
    "BEFORE": "p",
    "AFTER": "P",
    "INCLUDES": "o",
    "IS_INCLUDED": "O",
    "DURING": "d",
    "SIMULTANEOUS": "e",
    "IAFTER": "M",
    "IBEFORE": "m",
    "IDENTITY": "e",
    "BEGINS": "s",
    "ENDS": "f",
    "BEGUN_BY": "S",
    "ENDED_BY": "F",
    "DURING_INV": "D",
}

def composition(a_rel_b: str, b_rel_c: str) -> str:
    return composition_table[uci_rels.index(a_rel_b)][uci_rels.index(b_rel_c)]

def converse(rel: str) -> Optional[str]:
    return converse_table[rel] if rel in converse_table else None

def infer_relation(context: TimeInterval, rhs: IntervalVar) -> Set[str]:
    # create temp_graph
    vertices = list(context.get_interval_variables())

    # not in graph
    if rhs not in vertices:
        return set()

    final_rels = set()
    for relations in context.allen_relations_list:
        if not relations:
            continue

        G = nx.DiGraph()
        G.add_nodes_from(vertices)
        G.add_edges_from([(relation.lhs, relation.rhs) for relation in relations])

        graph = {k:{} for k in vertices}
        for relation in relations:
            graph[relation.lhs][relation.rhs] = relation.rel
            
        paths = list(nx.all_simple_paths(G, source = context.intervalvar, target = rhs))

        rels = [graph[i[0]][i[1]] for i in paths]
        if rels:
            final_rels |= rel_composition(rels)

    return final_rels

def rel_composition(rel_list):
    lhs = set(rel_list[0])
    for i in range(1, len(rel_list)):
        rhs = set(rel_list[i])
        possible_pairs = itertools.product(lhs, rhs)
        possible_results = [composition(j[0],j[1]) for j in possible_pairs]
        lhs = set(sum([list(i) for i in possible_results],[]))
    return lhs

def timeml_to_uci(timemlrel: str) -> str:
    return timeml_to_uci_dict[timemlrel]