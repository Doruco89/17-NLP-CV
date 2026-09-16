from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional

from .vocab import Vocab


def vocab(
    ordered_dict: Dict,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> Vocab:
    r"""토큰->빈도 OrderedDict 로부터 Vocab 을 만듭니다.

    ordered_dict 에 들어온 순서가 그대로 인덱스 순서가 됩니다.
    specials 는 ordered_dict 안에 같은 토큰이 있어도 제거한 뒤,
    special_first 에 따라 맨 앞(또는 맨 뒤)에 주어진 순서대로 붙습니다.
    """
    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = [token for token, freq in ordered_dict.items() if freq >= min_freq]

    if special_first:
        tokens[0:0] = specials
    else:
        tokens.extend(specials)

    return Vocab(tokens)


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> Vocab:
    r"""토큰 리스트를 내놓는 이터레이터로부터 Vocab 을 만듭니다.

    정렬 기준은 원본과 동일하게 (빈도 내림차순, 그 다음 사전순 오름차순)
    이라서, 같은 데이터면 인덱스까지 원본 torchtext 와 일치합니다.
    """
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is None:
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
    else:
        assert len(specials) < max_tokens, "len(specials) >= max_tokens 이면 vocab 이 특수토큰뿐입니다."
        ordered_dict = OrderedDict(sorted_by_freq_tuples[: max_tokens - len(specials)])

    return vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
