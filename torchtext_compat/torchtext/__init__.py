"""torchtext 호환 shim (순수 파이썬).

왜 존재하는가
    torchtext 0.18.0 의 C++ 확장(libtorchtext.pyd)은 torch 2.3.0 에 맞춰
    컴파일돼 있어서, 이 프로젝트의 메인 환경(torch 2.7.1+cu128)에서는
    import 단계에 OSError: [WinError 127] 로 죽습니다.
    그렇다고 torch 를 2.3.0 으로 내리면 RTX 50 시리즈(Blackwell, sm_120)를
    지원하는 CUDA 빌드가 없어 GPU 를 통째로 잃습니다.

    그래서 책 7장이 실제로 쓰는 torchtext API 세 가지만 순수 파이썬으로
    다시 구현했습니다. 노트북의 import 문과 호출 방식은 그대로 두고,
    메인 .venv 커널(GPU)에서 실행할 수 있습니다.

제공하는 것 (원본과 동일한 시그니처 / 동작)
    torchtext.datasets.Multi30k
    torchtext.data.utils.get_tokenizer
    torchtext.vocab.build_vocab_from_iterator, vocab, Vocab

제공하지 않는 것
    그 밖의 데이터셋, 사전학습 임베딩(GloVe/FastText), transforms,
    models, nn, prototype, experimental. 필요해지면 그때 추가하세요.
"""

import os

__version__ = "0.18.0+shim"

_CACHE_DIR = os.path.expanduser(os.path.join("~", ".torchtext", "cache"))

from . import data, datasets, vocab  # noqa: E402,F401


def disable_torchtext_deprecation_warning():
    """원본 API 호환용 no-op. shim 은 애초에 경고를 띄우지 않습니다."""


__all__ = ["data", "datasets", "vocab", "disable_torchtext_deprecation_warning"]
