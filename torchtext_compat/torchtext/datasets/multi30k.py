import hashlib
import os
import tarfile
import urllib.request
from typing import List, Tuple, Union

from torch.utils.data import Dataset

from torchtext import _CACHE_DIR

# 원본 torchtext 0.18.0 이 쓰는 것과 동일한 미러/체크섬입니다.
# (statmt.org 원본 서버가 내려간 뒤 torchtext 가 갈아탄 주소)
URL = {
    "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}

# 원본에서는 변수명이 MD5 지만 값도 검증방식도 sha256 입니다.
SHA256 = {
    "train": "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e",
    "valid": "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c",
    "test": "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36",
}

_PREFIX = {"train": "train", "valid": "val", "test": "test"}

NUM_LINES = {"train": 29000, "valid": 1014, "test": 1000}

DATASET_NAME = "Multi30k"

_VALID_SPLITS = ("train", "valid", "test")


class _Multi30k(Dataset):
    """(src, tgt) 문장쌍 데이터셋.

    원본은 torchdata 의 DataPipe(IterableDataset)를 돌려주지만, 여기서는
    map-style Dataset 으로 만들었습니다. 그래서 원본처럼 여러 번 순회할 수
    있는 것은 물론, DataLoader(shuffle=True) 도 그냥 됩니다.
    (IterableDataset 은 shuffle=True 를 못 씁니다.)
    """

    def __init__(self, pairs: List[Tuple[str, str]], split: str, language_pair: Tuple[str, str]):
        self._pairs = pairs
        self.split = split
        self.language_pair = tuple(language_pair)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self._pairs[index]

    def __iter__(self):
        return iter(self._pairs)

    def __repr__(self) -> str:
        src, tgt = self.language_pair
        return f"Multi30k(split={self.split!r}, language_pair=({src!r}, {tgt!r}), num_lines={len(self)})"


def _download(split: str, root: str) -> str:
    """tar.gz 를 캐시에 받고 sha256 을 검증한 뒤 경로를 돌려줍니다."""
    dest = os.path.join(root, os.path.basename(URL[split]))
    if not os.path.exists(dest):
        os.makedirs(root, exist_ok=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(URL[split], tmp)
        os.replace(tmp, dest)

    digest = hashlib.sha256()
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256[split]:
        os.remove(dest)
        raise RuntimeError(
            f"Multi30k {split} 체크섬이 맞지 않아 파일을 지웠습니다. 다시 실행해 받으세요."
        )
    return dest


def _extract(split: str, root: str, lang: str) -> str:
    """압축을 풀고 `{prefix}.{lang}` 파일 경로를 돌려줍니다."""
    target = os.path.join(root, f"{_PREFIX[split]}.{lang}")
    if os.path.exists(target):
        return target

    archive = _download(split, root)
    wanted = f"{_PREFIX[split]}.{lang}"
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            name = os.path.basename(member.name)
            # test 아카이브에는 macOS 가 남긴 `._test.de` 같은 더미가 섞여 있습니다.
            if name.startswith("._") or name != wanted:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            with open(target, "wb") as out:
                out.write(extracted.read())
            return target
    raise FileNotFoundError(f"{archive} 안에서 {wanted} 를 찾지 못했습니다.")


def _read_lines(path: str) -> List[str]:
    """줄바꿈을 떼고 읽습니다. train/val 파일 끝의 빈 줄 하나는 버립니다."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _load_split(root: str, split: str, language_pair: Tuple[str, str]) -> _Multi30k:
    src_lines = _read_lines(_extract(split, root, language_pair[0]))
    tgt_lines = _read_lines(_extract(split, root, language_pair[1]))
    if len(src_lines) != len(tgt_lines):
        raise RuntimeError(
            f"Multi30k {split}: 원문 {len(src_lines)}줄 / 번역문 {len(tgt_lines)}줄 로 개수가 다릅니다."
        )
    if len(src_lines) != NUM_LINES[split]:
        raise RuntimeError(
            f"Multi30k {split}: {NUM_LINES[split]}줄이어야 하는데 {len(src_lines)}줄입니다. "
            f"{root} 안의 캐시를 지우고 다시 받아보세요."
        )
    return _Multi30k(list(zip(src_lines, tgt_lines)), split, language_pair)


def Multi30k(
    root: str = _CACHE_DIR,
    split: Union[Tuple[str], str] = _VALID_SPLITS,
    language_pair: Tuple[str, str] = ("de", "en"),
):
    """Multi30k 독-영 번역 데이터셋.

    split 별 문장 수: train 29000 / valid 1014 / test 1000

    Args:
        root: 캐시 디렉터리. 기본값 ``~/.torchtext/cache``
        split: "train" / "valid" / "test" 하나 또는 여러 개의 튜플.
               문자열 하나를 주면 데이터셋 하나를, 튜플을 주면 같은 순서의
               튜플을 돌려줍니다. 기본값은 세 split 전부입니다.
        language_pair: ("de", "en") 또는 ("en", "de")

    Returns:
        (원문, 번역문) 문자열 쌍을 내놓는 Dataset
    """
    assert len(language_pair) == 2, "language_pair 는 (원문, 번역문) 두 개여야 합니다."
    assert tuple(sorted(language_pair)) == ("de", "en"), (
        "language_pair 는 ('de','en') 또는 ('en','de') 만 됩니다."
    )

    # 원본의 @_create_dataset_directory 와 같은 경로 규칙
    dataset_root = os.path.join(root, "datasets", DATASET_NAME)
    os.makedirs(dataset_root, exist_ok=True)

    single = isinstance(split, str)
    splits = (split,) if single else tuple(split)
    for item in splits:
        if item not in _VALID_SPLITS:
            raise ValueError(f"split 은 {_VALID_SPLITS} 중에서 골라야 합니다. 받은 값: {item!r}")

    datasets = tuple(_load_split(dataset_root, item, tuple(language_pair)) for item in splits)
    return datasets[0] if single else datasets
