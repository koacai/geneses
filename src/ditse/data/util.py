from pathlib import Path


def glob_wds(dir: str) -> list[str]:
    wds_paths = []
    wds_paths.extend(list(map(str, Path(dir).glob("**/*.tar.gz"))))
    wds_paths.extend(list(map(str, Path(dir).glob("**/*.tar"))))
    return wds_paths
