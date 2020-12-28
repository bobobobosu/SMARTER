from typing import List, Dict, Any
from templi.models.temli_multiway_match import BertMultiwayMatch
from allennlp.data import Vocabulary
from allennlp.common import Registrable
from torch import LongTensor
import torch

class TemliQuestionEmbedder(Registrable):
    def __init__(self, bert_model, device) -> None:
        self._bert_multiway_match = BertMultiwayMatch.from_pretrained(bert_model)
        self._bert_multiway_match.to(device=device)

    def get_output_dim(self) -> int:
        return self._bert_multiway_match.config.hidden_size

    def __call__(self, question: LongTensor, metadata: List[Dict[str, Any]]):
        # question: (batch_size, arg_idx, doc_len)
        # multiway_arg: (arg_idx, batch_size, doc_len)
        multiway_arg = torch.transpose(question, 0, 1)
        input_ids = multiway_arg[0]
        token_type_ids = multiway_arg[1]
        attention_mask = multiway_arg[2]
        doc_len = LongTensor([i["doc_len"] for i in metadata]).to(question.device)
        ques_len = LongTensor([i["ques_len"] for i in metadata]).to(question.device)
        return self._bert_multiway_match.forward(
            input_ids, token_type_ids, attention_mask, doc_len, ques_len
        )
