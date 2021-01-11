from allennlp.data.tokenizers.tokenizer import Tokenizer
from templi.preprocessors.tokenization import BertTokenizer
from overrides import overrides
from typing import List, Optional
from allennlp.data.tokenizers.tokenizer import Token


class TemliTokenizer(Tokenizer):
    def __init__(self, bert_model, do_lower_case) -> None:
        self._tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=do_lower_case
        )

    # TODO Implement def batch_tokenize(self, texts: List[str]) -> List[List[Token]]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # TODO ERROR: Vocabulary tokens must be strings, or saving and loading will break.
        tokes = self._tokenizer.tokenize(text)
        ids = self._tokenizer.convert_tokens_to_ids(tokes)
        return [Token(tokes[i], text_id=ids[i]) for i in range(len(tokes))]

    def tokenize_to_str(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self._tokenizer.convert_tokens_to_ids(tokens)
