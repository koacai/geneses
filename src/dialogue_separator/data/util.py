from pathlib import Path
from typing import Sequence


def glob_wds(paths: Sequence[str]) -> list[str]:
    wds_paths = []
    for path in paths:
        wds_paths.extend(list(map(str, Path(path).glob("**/*.tar.gz"))))
    for path in paths:
        wds_paths.extend(list(map(str, Path(path).glob("**/*.tar"))))
    return wds_paths
