from typing import Dict, List, NamedTuple, Set, Type, Tuple, Any
from allennlp_semparse.fields.knowledge_graph_field import KnowledgeGraphField
from allennlp_semparse.common.knowledge_graph import KnowledgeGraph
from collections import defaultdict


def hard_coded_knowledge_graph():
    neighbors = {}
    return KnowledgeGraph(
        set(neighbors.keys()), neighbors
    )  # TODO fill in entity_text key


class TempliTimeContext:
    """
    Create a context for this specific question

    Parameters
    ----------
    temp_vars : ``Dict[str:str]``
        All temporal annotations (including time and event) in the sentence
        The key is the spaceless_rng of the variable and value is the original string

    """

    def __init__(
        self, temp_vars: Dict[str, str], knowledge_graph: Dict[str, List[str]]
    ) -> None:
        self.temp_vars = temp_vars
        self.interval_types: Set[str] = set()
        self._table_knowledge_graph = KnowledgeGraph(
            set(knowledge_graph.keys()),
            knowledge_graph,
            {k: k for k in knowledge_graph.keys()},
        )

    def get_table_knowledge_graph(self) -> KnowledgeGraph:
        if self._table_knowledge_graph is None:
            self._table_knowledge_graph = hard_coded_knowledge_graph()
        return self._table_knowledge_graph
