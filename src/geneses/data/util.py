from pathlib import Path

import numpy as np


def glob_wds(dir: str) -> list[str]:
    wds_paths = []
    wds_paths.extend(list(map(str, Path(dir).glob("**/*.tar.gz"))))
    wds_paths.extend(list(map(str, Path(dir).glob("**/*.tar"))))
    return wds_paths


def glob_wds_split(dir: str, split_idx: int, array_num: int) -> list[str]:
    """Get a subset of shards for parallel processing.

    Args:
        dir: Directory containing the shards
        split_idx: Index of the current job (0 to array_num-1)
        array_num: Total number of parallel jobs

    Returns:
        List of shard paths for this job
    """
    all_shards = glob_wds(dir)
    all_shards = sorted(all_shards)  # Ensure consistent ordering

    # Split shards across jobs
    shards_per_job = len(all_shards) // array_num
    remainder = len(all_shards) % array_num

    # Calculate start and end indices for this job
    if split_idx < remainder:
        # First 'remainder' jobs get one extra shard
        start_idx = split_idx * (shards_per_job + 1)
        end_idx = start_idx + shards_per_job + 1
    else:
        start_idx = split_idx * shards_per_job + remainder
        end_idx = start_idx + shards_per_job

    return all_shards[start_idx:end_idx]


def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min([get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h])

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[: max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample
