from __future__ import annotations

from pathlib import Path


class IDSDictionary:
    """Inference-time IDS vocabulary loader for the TextVerifier checkpoint."""

    def __init__(self) -> None:
        self.char_to_ids: dict[str, str] = {}
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    @classmethod
    def load(cls, data_dir: str | Path) -> 'IDSDictionary':
        data_dir = Path(data_dir)
        instance = cls()
        for line in (data_dir / 'ids_decompose.txt').read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            char, ids = line.split(':', 1)
            instance.char_to_ids[char] = ids
        for index, token in enumerate((data_dir / 'ids_vocab.txt').read_text(encoding='utf-8').splitlines()):
            instance.token2id[token] = index
            instance.id2token[index] = token
        if not instance.token2id:
            raise ValueError(f'empty IDS vocabulary: {data_dir}')
        return instance
