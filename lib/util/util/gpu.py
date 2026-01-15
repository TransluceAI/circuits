import os
import subprocess

import torch


def gpu_mem_str() -> str:
    device_count = torch.cuda.device_count()
    return "\n".join(
        [
            "Allocated: "
            + " | ".join(
                [
                    f"{torch.cuda.memory_allocated(f'cuda:{i}') / 1e9:.2f} GB"
                    for i in range(device_count)
                ]
            ),
            "Reserved: "
            + " | ".join(
                [
                    f"{torch.cuda.memory_reserved(f'cuda:{i}') / 1e9:.2f} GB"
                    for i in range(device_count)
                ]
            ),
        ]
    )


def list_free_gpus(thresh: int = 100) -> list[int]:
    # Get the GPU ids by checking CUDA_VISIBLE_DEVICES.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_pool = [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        gpu_pool = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        used_mems = result.stdout.strip().split("\n")
        if gpu_pool is None:
            free_gpus = [idx for idx, used_mem in enumerate(used_mems) if float(used_mem) <= thresh]
        else:
            free_gpus = [
                idx
                for idx, used_mem in enumerate(used_mems)
                if float(used_mem) <= thresh and idx in gpu_pool
            ]

        return free_gpus
    except subprocess.CalledProcessError:
        # nvidia-smi command failed
        return []
    except FileNotFoundError:
        # nvidia-smi not found
        return []
