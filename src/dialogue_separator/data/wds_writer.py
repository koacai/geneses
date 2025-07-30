import os
import pathlib
import uuid
from multiprocessing import Manager, Process, Queue

import torch
import tqdm
import webdataset as wds
from omegaconf import DictConfig


def process_item(item):
    """
    Processes a single data item, converting its contents into a format
    suitable for WebDataset, with specific file extensions based on type.
    """
    output_item = dict()
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            output_item[k + ".pth"] = v
        elif isinstance(v, str):
            output_item[k + ".txt"] = v
        elif isinstance(v, int):
            output_item[k + ".cls"] = v
        else:
            output_item[k + ".pickle"] = v
    output_item["__key__"] = uuid.uuid4().hex
    return output_item


def writer_process(worker_id: int, queue: Queue, cfg: DictConfig, output_dir: str):
    """
    A worker process that consumes data from a queue and writes it to its own set of shards.

    Args:
        worker_id (int): A unique ID for this worker to ensure output files are unique.
        queue (Queue): The shared queue from which to pull data items.
        cfg (DictConfig): The Hydra configuration object.
        output_dir (str): The directory where shards will be saved.
    """
    shard_pattern = os.path.join(output_dir, f"worker-{worker_id}-dataset-%06d.tar")

    with wds.ShardWriter(
        shard_pattern, maxcount=cfg.data.get("shard_maxcount", 1000)
    ) as sink:
        while True:
            item = queue.get()

            if item == "DONE":
                break

            processed = process_item(item)
            sink.write(processed)


def run_parallel_writing(
    dataloader: wds.WebLoader,
    output_dir: str,
    num_writers: int,
    cfg: DictConfig,
):
    """
    Orchestrates the parallel writing process for a given dataloader.

    Args:
        dataloader (DataLoader): The dataloader to source data from.
        output_dir (str): The target directory for the output shards.
        num_writers (int): The number of parallel writer processes to spawn.
        cfg (DictConfig): The Hydra configuration object.
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with Manager() as manager:
        data_queue = manager.Queue(maxsize=num_writers * 40)

        processes = []
        for i in range(num_writers):
            p = Process(target=writer_process, args=(i, data_queue, cfg, output_dir))
            processes.append(p)
            p.start()

        for item in tqdm.tqdm(dataloader, desc=f"Writing to {output_dir}"):
            data_queue.put(item)

        print("Finished feeding queue. Sending termination signals...")
        for _ in range(num_writers):
            data_queue.put("DONE")

        for p in processes:
            p.join()

        print(f"All writer processes for '{output_dir}' have finished.")
