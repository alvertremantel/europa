from __future__ import annotations


class VisualizerBase:
    def __init__(self, tokenizer_vocab: dict[int, str] | None = None) -> None:
        self.tokenizer_vocab = tokenizer_vocab or {}
        self.fig_count = 0

    def _token_str(self, token_id: int) -> str:
        if token_id in self.tokenizer_vocab:
            return self.tokenizer_vocab[token_id]
        return f"[{token_id}]"
