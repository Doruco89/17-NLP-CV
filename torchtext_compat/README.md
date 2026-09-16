# torchtext 호환 shim

책 7장(Transformer 번역)의 torchtext 코드를 **메인 `.venv`(GPU)에서 그대로** 돌리기 위한
순수 파이썬 재구현입니다. 노트북의 import 문과 호출 방식은 바꾸지 않습니다.

```python
from torchtext.datasets import Multi30k          # 그대로 동작
from torchtext.data.utils import get_tokenizer   # 그대로 동작
from torchtext.vocab import build_vocab_from_iterator
```

## 왜 필요한가

| | 메인 `.venv` | `.venv-torchtext` |
|---|---|---|
| torch | 2.7.1+cu128 | 2.3.0+cpu |
| `cuda.is_available()` | **True** (RTX 5060) | False |
| 진짜 torchtext | 못 씀 | 씀 (단, Multi30k는 고장) |

두 가지가 동시에 막혀 있었습니다.

1. **torchtext 0.18 + torch 2.7** → `import torchtext` 단계에서
   `OSError: [WinError 127]`. 휠에 들어있는 C++ 확장 `libtorchtext.pyd` 가
   torch 2.3 ABI 에 링크돼 있어서 로드되지 않습니다.
2. **torch 2.3 으로 내리기** → RTX 50 시리즈(Blackwell, `sm_120`)를 지원하는
   CUDA 빌드가 torch 2.3 에는 없습니다. GPU 를 통째로 잃습니다.

덤으로, `.venv-torchtext` 의 진짜 torchtext 도 `Multi30k` 는 못 씁니다.
`torchdata 0.7.1` 이 torch 2.3 에서 제거된 `DILL_AVAILABLE` 을 import 하다
`ImportError` 로 죽습니다. `vocab` / `get_tokenizer` 만 살아 있습니다.

## 어떻게 붙어 있나

`.venv/Lib/site-packages/torchtext_shim.pth` 가 이 `torchtext_compat/` 폴더를 `sys.path` 에
추가합니다. 메인 `.venv` 에만 적용되고 `.venv-torchtext` 는 건드리지 않습니다.

**`.venv` 를 새로 만들면 이 `.pth` 가 사라집니다.** 그때는 이렇게 복구하세요.

```bash
python -c "import os;p=os.path.abspath('torchtext_compat');open(os.path.join('.venv','Lib','site-packages','torchtext_shim.pth'),'w',encoding='utf-8').write(p+'\n')"
python -m pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl
```

### 에디터(Pylance)가 빨간/노란 줄을 그을 때

`.pth` 는 실행 중에만 먹히고 정적 분석기는 읽지 못합니다. 그래서 셀은 멀쩡히
돌아가는데 `from torchtext.datasets import Multi30k` 밑에 "import 를 확인할 수
없습니다" 경고가 뜹니다. 워크스페이스에 `.vscode/settings.json` 을 만들어
shim 폴더를 알려주세요. (이 파일은 `.gitignore` 에 있으므로 클론할 때마다 필요합니다.)

```json
{
  "python.analysis.extraPaths": ["./torchtext_compat"],
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
```

저장한 뒤 `Ctrl+Shift+P` → **Developer: Reload Window**. 경고가 사라지고
`Multi30k` / `build_vocab_from_iterator` 자동완성도 같이 살아납니다.

## 구현 범위

제공: `datasets.Multi30k`, `data.utils.get_tokenizer`,
`vocab.build_vocab_from_iterator`, `vocab.vocab`, `vocab.Vocab`

미제공: 그 밖의 데이터셋, 사전학습 임베딩(GloVe/FastText), `transforms`,
`models`, `nn`, `prototype`, `experimental`, 그리고 moses/toktok/revtok/subword
토크나이저. 필요해지면 그때 추가하세요 — 없는 걸 부르면 조용히 틀리지 않고
명시적으로 실패합니다.

### 원본과 다른 점 (의도한 것)

- `Multi30k` 가 torchdata DataPipe 대신 **map-style `Dataset`** 을 돌려줍니다.
  원본처럼 여러 번 순회할 수 있고, 추가로 `len()`, 인덱싱,
  `DataLoader(shuffle=True)` 가 됩니다. (원본의 `IterableDataset` 은
  `shuffle=True` 를 못 씁니다.)
- 원본은 `.shuffle()` 이 datapipe 에 박혀 있지만 여기선 없습니다.
  섞는 건 `DataLoader(shuffle=True)` 로 하세요.
- 데이터 파일 끝의 빈 줄을 버려서 split 크기가 torchtext 가 문서화한
  29000 / 1014 / 1000 과 정확히 맞습니다.

## 원본과 동일함을 검증한 방법

같은 스크립트를 양쪽 환경(shim / 진짜 torchtext 0.18.0)에서 돌려
`build_vocab_from_iterator` 결과를 비교했습니다. `min_freq` 1·2,
`special_first` 앞·뒤, 공백/`basic_english` 토크나이저 조합에 대해
**vocab 크기, itos 전체의 sha256, 토큰→인덱스 변환 결과, OOV 처리까지
29개 항목 전부 일치**했습니다.

데이터는 원본 torchtext 와 같은 미러 URL 에서 받고 같은 sha256 으로 검증합니다.
캐시 위치도 원본과 같은 `~/.torchtext/cache/datasets/Multi30k/` 입니다.
