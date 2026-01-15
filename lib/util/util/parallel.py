from __future__ import annotations

import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm


class TensorDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check that all values are tensors
        for k, v in self.items():
            assert isinstance(
                v, torch.Tensor
            ), f"Value for key '{k}' must be a torch.Tensor, but got {type(v)}"

    @property
    def shapes(self):
        return {k: v.shape for k, v in self.items()}

    @property
    def devices(self):
        return {k: v.device for k, v in self.items()}

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self.values())

    def to(self, device: str):
        return TensorDict({k: v.to(device) for k, v in self.items()})

    def is_shared(self):
        return all(v.is_shared() for v in self.values())

    def select(self, dim: int, index: int):
        return TensorDict({k: v.select(dim, index) for k, v in self.items()})

    def narrow(self, dim: int, start: int, length: int):
        return TensorDict({k: v.narrow(dim, start, length) for k, v in self.items()})

    def copy_(self, other: TensorDict):
        assert isinstance(other, TensorDict), "Cannot copy from a non-TensorDict"
        for k, v in self.items():
            v.copy_(other[k])


def parallelize_work(
    shared_inputs: Dict[str, torch.Tensor],
    batch_inputs: Dict[str, torch.Tensor],
    shared_batch_inputs: Dict[str, torch.Tensor],
    shared_batch_outputs: Dict[str, torch.Tensor],
    batch_size: int,
    num_batches: int,
    work_fn: Callable[[Dict[str, torch.Tensor], int, str], Dict[str, torch.Tensor]],
    handle_output_fn: Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int], None],
    use_gpus: List[int] = list(range(torch.cuda.device_count())),
):
    """
    This function parallelizes work over multiple GPUs.
    We play a bunch of tricks to minimize IPC costs of large tensors being moved across GPUs.

    Let:
        - N be the total number of samples
        - B be the batch size
        - G is the number of GPUs

    What you provide:
        - A list of fixed inputs that all tasks need
        - A list of variable inputs, each of which is (N, ...)
        - A list of shared batch input tensors for GPU IPC, each of which is (G, B, ...)
        - A list of shared batch output tensors for GPU IPC, each of which is (G, B, ...)

    How this works:
        - Each variable input is broken into batches of size B
        - Each batch is assigned a GPU rank
        - It is then placed onto the shared batch input IPC tensor at its GPU rank
        - The work function is called for each batch on the corresponding GPU
        - The output of that work function is placed onto the shared batch output IPC tensor at its GPU rank
        - We then call `handle_output_fn`
    """

    mp.set_start_method("spawn", force=True)

    # Check that all IPC memory is indeed shared
    assert all(
        tensor.is_shared()
        for tensor in shared_inputs.values()
        if isinstance(tensor, torch.Tensor) or isinstance(tensor, TensorDict)
    ), "All shared input tensors must be shared"
    assert all(
        tensor.is_shared() for tensor in shared_batch_inputs.values()
    ), "All shared batch input tensors must be shared"
    assert all(
        tensor.is_shared() for tensor in shared_batch_outputs.values()
    ), "All shared batch output tensors must be shared"

    # Input read/write synchronization
    data_requests = [mp.Event() for _ in range(max(use_gpus) + 1)]
    data_readys = [mp.Event() for _ in range(max(use_gpus) + 1)]

    # Output read/write synchronization
    ready_to_reads = [mp.Event() for _ in range(max(use_gpus) + 1)]
    ready_to_writes = [mp.Event() for _ in range(max(use_gpus) + 1)]
    for event in ready_to_writes:
        event.set()  # Allow writing to the memory at first

    # Task indices
    input_batch_indices = [mp.Value("i", -1) for _ in range(max(use_gpus) + 1)]
    output_batch_indices = [mp.Value("i", -1) for _ in range(max(use_gpus) + 1)]

    # Task queue
    task_queue = mp.Queue()
    for batch_idx in range(num_batches):
        task_queue.put({"batch_idx": batch_idx})

    # Wrap computation with a try-catch to terminate process (and thus free memory) in case of failure
    try:
        # Start processes
        processes = []
        for device_rank in use_gpus:
            p = mp.Process(
                target=worker_function,
                args=(
                    task_queue,
                    work_fn,
                    # Data
                    {
                        k: (
                            # Shared inputs should be moved to the correct device if possible
                            # CUDA IPC is faster than CPU IPC (I think)
                            v.to(f"cuda:{device_rank}")
                            if isinstance(v, TensorDict) or isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in shared_inputs.items()
                    },
                    shared_batch_inputs,
                    shared_batch_outputs,
                    # Input sync
                    data_requests[device_rank],
                    data_readys[device_rank],
                    input_batch_indices[device_rank],
                    # Output sync
                    ready_to_reads[device_rank],
                    ready_to_writes[device_rank],
                    output_batch_indices[device_rank],
                    # Device
                    f"cuda:{device_rank}",
                    device_rank,
                ),
            )
            p.start()
            processes.append(p)

        pbar = tqdm(total=num_batches * batch_size, desc="Computing stuff")

        # Fn to handle R/W for each GPU
        def _manage_gpu_tasks(gpu_id: int):
            print(f"Starting I/O thread for GPU {gpu_id}")
            while True:
                # Wait for a data request to come in
                data_requests[gpu_id].wait()
                data_requests[gpu_id].clear()

                # Copy the requested data into the shared tensors
                batch_idx = input_batch_indices[gpu_id].value
                if batch_idx == -100:
                    print(f"GPU {gpu_id} got -100, exiting...")
                    break

                # Copy the input data to the shared tensors
                # The order must be consistent
                for batch_input, shared_batch_input in zip(
                    batch_inputs.values(), shared_batch_inputs.values()
                ):
                    # print(shared_batch_input.shape)
                    # print(batch_input.narrow(0, batch_idx * batch_size, batch_size).shape)
                    shared_batch_input.select(0, gpu_id).copy_(
                        batch_input.narrow(0, batch_idx * batch_size, batch_size)
                    )

                # Indicate that data is ready
                data_readys[gpu_id].set()

                # Block until it's okay to read
                ready_to_reads[gpu_id].wait()
                ready_to_reads[gpu_id].clear()

                # Get the result
                batch_idx = output_batch_indices[gpu_id].value
                handle_output_fn(
                    shared_inputs,
                    {k: v.select(0, gpu_id) for k, v in shared_batch_inputs.items()},
                    {k: v.select(0, gpu_id) for k, v in shared_batch_outputs.items()},
                    batch_idx,
                )

                # Signal that writing can resume
                ready_to_writes[gpu_id].set()
                pbar.update(batch_size)

        # Start threads to listen for results and collect futures
        futures = []
        with ThreadPoolExecutor(max_workers=len(use_gpus)) as executor:
            for i in use_gpus:
                future = executor.submit(_manage_gpu_tasks, i)
                futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            future.result()
        print("All threads closed")

        # Wait for all processes to finish
        for p in processes:
            p.join()
        print("All processes finished")

    # Kill all processes on exception
    except Exception as e:
        print(f"Caught exception: {e}")
        traceback.print_exc()

        for p in processes:
            p.terminate()


def worker_function(
    task_queue: mp.Queue,
    work_fn: Callable[[Dict[str, torch.Tensor], int, str], Dict[str, torch.Tensor]],
    shared_inputs: Dict[str, torch.Tensor],
    shared_batch_inputs: Dict[str, torch.Tensor],
    shared_batch_outputs: Dict[str, torch.Tensor],
    data_request: mp.Event,
    data_ready: mp.Event,
    input_batch_index: mp.Value,
    ready_to_read: mp.Event,
    ready_to_write: mp.Event,
    output_batch_index: mp.Value,
    device: str | torch.device,
    device_rank: int,
):
    try:
        while task_queue.qsize() > 0:
            qpop_done = False
            while not qpop_done:
                try:
                    task = task_queue.get_nowait()
                    qpop_done = True
                    task_done = False
                except queue.Empty:
                    if task_queue.qsize() == 0:
                        qpop_done = True
                    else:
                        print(
                            f"WARN: GPU {device} encountered queue empty with {task_queue.qsize()} tasks left. Retrying..."
                        )
                        time.sleep(0.1)

            # Request tensors
            batch_idx = task["batch_idx"]
            input_batch_index.value = batch_idx
            data_request.set()

            # Block until data is ready
            data_ready.wait()
            data_ready.clear()

            # Do work
            res = work_fn(
                shared_inputs,
                {k: v.select(0, device_rank) for k, v in shared_batch_inputs.items()},
                batch_idx,
                device,
            )

            # Block until it's okay to write
            ready_to_write.wait()
            ready_to_write.clear()
            output_batch_index.value = batch_idx

            # Write
            for k, v in res.items():
                shared_batch_outputs[k].select(0, device_rank).copy_(v)

            # Indicate that we are done
            ready_to_read.set()
            task_done = True

    # Catch exceptions and print traceback as machine dies
    except Exception as e:
        print(f"GPU {device} encountered error: {e}")
        traceback.print_exc()

        # If there is a task and it's not done, push it back on the queue
        if task_done == False and task is not None:
            print("Pushing failed task back on the queue")
            task_queue.put(task)

    # Indicate to master process that we are done w/ -100
    finally:
        print(f"GPU {device} finished; writing -100 to CPU memory")
        input_batch_index.value = -100
        data_request.set()

        ready_to_write.wait()
        ready_to_write.clear()
        output_batch_index.value = -100
        ready_to_read.set()
