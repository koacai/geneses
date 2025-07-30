import os
import pathlib
import uuid
from multiprocessing import Manager, Process, Queue

import torch
import tqdm
import webdataset as wds


def process_item(item):
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


def writer_process(worker_id: int, queue: Queue, output_dir: str, shard_maxcount: int):
    shard_pattern = os.path.join(output_dir, f"worker-{worker_id}-dataset-%06d.tar")

    with wds.ShardWriter(shard_pattern, maxcount=shard_maxcount) as sink:
        while True:
            item = queue.get()

            if item == "DONE":
                break

            processed = process_item(item)
            sink.write(processed)


def run_parallel_writing(
    dataloader: wds.WebLoader, output_dir: str, num_writers: int, shard_maxcount: int
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with Manager() as manager:
        data_queue = manager.Queue(maxsize=num_writers * 40)

        processes = []
        for i in range(num_writers):
            p = Process(
                target=writer_process, args=(i, data_queue, output_dir, shard_maxcount)
            )
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
