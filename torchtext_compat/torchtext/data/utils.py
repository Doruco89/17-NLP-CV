import re
from functools import partial

_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
_patterns_dict = [(re.compile(p), r) for p, r in zip(_patterns, _replacements)]


def _split_tokenizer(x):
    return x.split()


def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]


def _basic_english_normalize(line):
    r"""소문자화 + 기본 문장부호 분리 후 공백 기준 split. 원본과 동일한 규칙입니다."""
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def get_tokenizer(tokenizer, language="en"):
    r"""문장 -> 토큰 리스트 함수를 돌려줍니다.

    지원하는 값
        None            : str.split()
        "basic_english" : 소문자화 + 문장부호 분리 (language 는 "en" 만 허용)
        "spacy"         : spacy.load(language) 의 토크나이저
                          예) get_tokenizer("spacy", language="de_core_news_sm")
        callable        : 그대로 반환

    원본의 moses / toktok / revtok / subword 는 각각 별도 패키지가 필요해서
    이 shim 에서는 빼뒀습니다. 쓰려고 하면 명시적으로 실패합니다.
    """
    if tokenizer is None:
        return _split_tokenizer

    if tokenizer == "basic_english":
        if language != "en":
            raise ValueError("basic_english 정규화는 영어(en)에만 쓸 수 있습니다.")
        return _basic_english_normalize

    if callable(tokenizer):
        return tokenizer

    if tokenizer == "spacy":
        import spacy

        try:
            nlp = spacy.load(language)
        except OSError as exc:
            raise OSError(
                f"spacy 모델 {language!r} 을(를) 찾을 수 없습니다.\n"
                f"  python -m spacy download {language}\n"
                "또는 해당 모델 휠을 pip 로 설치하세요."
            ) from exc
        return partial(_spacy_tokenize, spacy=nlp)

    raise NotImplementedError(
        f"{tokenizer!r} 토크나이저는 이 torchtext shim 에 구현돼 있지 않습니다. "
        "None / 'basic_english' / 'spacy' / 직접 만든 함수 중에서 고르세요."
    )
