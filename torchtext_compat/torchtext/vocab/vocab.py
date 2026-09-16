from typing import Dict, List, Optional

import torch.nn as nn


class Vocab(nn.Module):
    r"""토큰 <-> 인덱스 매핑. 원본 torchtext.vocab.Vocab 의 순수 파이썬 재구현.

    원본은 C++ VocabPybind 를 감싸지만, 여기서는 list/dict 로 같은 동작을 냅니다.
    기본 인덱스(default index)가 설정돼 있지 않은 상태에서 OOV 토큰을 조회하면
    원본과 동일하게 RuntimeError 를 냅니다.
    """

    def __init__(self, tokens: List[str], default_index: Optional[int] = None) -> None:
        super().__init__()
        self._itos: List[str] = list(tokens)
        self._stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self._itos)}
        if len(self._stoi) != len(self._itos):
            raise ValueError("중복된 토큰이 있습니다. vocab 은 중복을 허용하지 않습니다.")
        self._default_index = default_index

    # --- 조회 ---------------------------------------------------------------
    def forward(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __getitem__(self, token: str) -> int:
        index = self._stoi.get(token)
        if index is not None:
            return index
        if self._default_index is None:
            raise RuntimeError(
                f"토큰 {token!r} 이(가) vocab 에 없고 기본 인덱스도 설정되지 않았습니다. "
                "vocab.set_default_index(vocab['<unk>']) 를 먼저 호출하세요."
            )
        return self._default_index

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def lookup_token(self, index: int) -> str:
        if not 0 <= index < len(self._itos):
            raise RuntimeError(f"인덱스 {index} 가 범위 [0, {len(self._itos)}) 를 벗어났습니다.")
        return self._itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.lookup_token(index) for index in indices]

    # --- 기본 인덱스 --------------------------------------------------------
    def set_default_index(self, index: Optional[int]) -> None:
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    # --- 수정 ---------------------------------------------------------------
    def insert_token(self, token: str, index: int) -> None:
        if token in self._stoi:
            raise RuntimeError(f"토큰 {token!r} 이(가) 이미 vocab 에 있습니다.")
        if not 0 <= index <= len(self._itos):
            raise RuntimeError(f"인덱스 {index} 가 범위 [0, {len(self._itos)}] 를 벗어났습니다.")
        self._itos.insert(index, token)
        self._stoi = {tok: i for i, tok in enumerate(self._itos)}

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self._itos))

    # --- 내보내기 -----------------------------------------------------------
    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        return list(self._itos)
