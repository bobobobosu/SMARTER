from typing import Dict, List, NamedTuple, Set, Type, Tuple, Any


class TempliTimeContext:
    def __init__(self, temp_vars: Set[str]) -> None:
        self.temp_vars = temp_vars
