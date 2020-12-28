from templi.models.temli_multiway_match import BertEmbeddings
from allennlp.common import Registrable

class TemliWordEmbedder(Registrable):
    def __init__(self, bert_model) -> None:
        # self._tokenizer = BertEmbeddings(bert_model)
        pass
    
    # def forward()