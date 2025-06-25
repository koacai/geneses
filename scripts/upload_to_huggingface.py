import json
import os
from pathlib import Path

from huggingface_hub import HfApi

if __name__ == "__main__":
    ckpt_path = "dialogue-separator/rzjrr29f/checkpoints/epoch=112-step=57404.ckpt"
    config_path = "outputs/2025-06-23/14-18-00/.hydra/config.yaml"
    metadata_path = "wandb/run-20250623_141828-rzjrr29f/files/wandb-metadata.json"

    keys_to_extract_from_metadata = [
        "os",
        "python",
        "startedAt",
        "codePath",
        "git",
        "cpu_count",
        "cpu_count_logical",
        "gpu",
        "gpu_count",
        "cpu",
        "gpu_nvidia",
        "cudaVersion",
    ]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata_filtered = {
        k: metadata[k] for k in keys_to_extract_from_metadata if k in metadata
    }

    tmp_metadata_path = "tmp_metadata.json"
    with open(tmp_metadata_path, "w") as f:
        json.dump(metadata_filtered, f, indent=2)

    dir = "Libri2Mix/L1Loss"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=f"{dir}/{Path(ckpt_path).name}",
        repo_id="koacai/dialogue-separator",
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo=f"{dir}/config.yaml",
        repo_id="koacai/dialogue-separator",
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=tmp_metadata_path,
        path_in_repo=f"{dir}/metadata.json",
        repo_id="koacai/dialogue-separator",
        repo_type="model",
    )

    os.remove(tmp_metadata_path)
