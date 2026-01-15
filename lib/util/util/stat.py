from typing import Dict, List

import torch
from util.malloc import malloc_cpu_if_None
from util.parallel import parallelize_work


class ParallelBatchCovariance:
    """
    Running computation. Use this when the entire covariance matrix is needed,
    and when the whole covariance matrix fits in the GPU.
    Chan-style numerically stable update of mean and full covariance matrix.
    Chan, Golub. LeVeque. 1983. http://www.jstor.org/stable/2683386
    """

    def __init__(self, n_matrices: int, block_size: int, var_dim: int):
        """
        Update batch size must always be the same due to page locking stuff.
        """

        assert n_matrices % block_size == 0, "n_matrices must be a multiple of update_batch_size"

        self._var_dim = var_dim
        self._n_blocks = int(n_matrices / block_size)
        self._block_size = block_size

        self._initialized_B = torch.zeros(self._n_blocks, device="cpu", dtype=torch.bool)
        self._counts_BN = torch.zeros(
            self._n_blocks, self._block_size, device="cpu", dtype=torch.float32
        )
        self._means_BND = torch.zeros(
            self._n_blocks,
            self._block_size,
            self._var_dim,
            device="cpu",
            dtype=torch.float32,
        )
        self._cmom2s_BNDD = torch.zeros(
            self._n_blocks,
            self._block_size,
            self._var_dim,
            self._var_dim,
            device="cpu",
            dtype=torch.float32,
        )

    @torch.no_grad()
    def covariance(self, matrix_idx: int, unbiased=True):
        block_idx, matrix_idx_in_block = divmod(matrix_idx, self._block_size)
        return self._cmom2s_BNDD[block_idx][matrix_idx_in_block] / (
            self._counts_BN[block_idx][matrix_idx_in_block] - (1 if unbiased else 0)
        )


def acc_covariance(
    covs: ParallelBatchCovariance,
    dirs_NXXXD: torch.Tensor,
    shared_dirs_GBXXXD: torch.Tensor | None = None,
    shared_counts_GB: torch.Tensor | None = None,
    shared_means_GBD: torch.Tensor | None = None,
    shared_cmom2s_GBDD: torch.Tensor | None = None,
    use_gpus: List[int] = list(range(torch.cuda.device_count())),
):
    B, G, D = covs._block_size, max(use_gpus) + 1, dirs_NXXXD.size(-1)

    ##########
    # Malloc #
    ##########

    _m = malloc_cpu_if_None

    print(f"Allocating shared memory... ", end="", flush=True)
    initialized_GB = _m(None, (G, B), torch.bool, shared=True)

    print("dirs... ", end="", flush=True)
    shared_dirs_GBXXXD = _m(
        shared_dirs_GBXXXD, (G, B, *dirs_NXXXD.shape[1:]), torch.float32, shared=True
    )

    print("cov... ", end="", flush=True)
    shared_counts_GB = _m(shared_counts_GB, (G, B), torch.float32, shared=True)
    shared_means_GBD = _m(shared_means_GBD, (G, B, D), torch.float32, shared=True)
    shared_cmom2s_GBDD = _m(shared_cmom2s_GBDD, (G, B, D, D), torch.float32, shared=True)

    print("Done")

    ###########
    # Execute #
    ###########

    parallelize_work(
        shared_inputs={},
        batch_inputs={
            "dirs_NXXXD": dirs_NXXXD,
            "initialized_N": covs._initialized_B.repeat_interleave(B),
            "counts_N": covs._counts_BN.view(-1),
            "means_ND": covs._means_BND.view(-1, D),
            "cmom2s_NDD": covs._cmom2s_BNDD.view(-1, D, D),
        },
        shared_batch_inputs={
            "shared_dirs_SXXXD": shared_dirs_GBXXXD,
            "initialized_S": initialized_GB,
            "shared_mem_S": shared_counts_GB,
            "shared_mem_SD": shared_means_GBD,
            "shared_mem_SDD": shared_cmom2s_GBDD,
        },
        shared_batch_outputs={},
        batch_size=B,
        num_batches=covs._n_blocks,
        work_fn=_work_fn,
        handle_output_fn=lambda _, shared_batch_inputs, shared_batch_outputs, batch_idx: handle_output_fn(
            shared_batch_inputs, shared_batch_outputs, batch_idx, covs
        ),
        use_gpus=use_gpus,
    )


def _work_fn(_, shared_batch_inputs: Dict[str, torch.Tensor], batch_idx: int, device: str):
    # Gather inputs
    shared_dirs_SXXXD = shared_batch_inputs["shared_dirs_SXXXD"]
    initialized_S, shared_mem_S, shared_mem_SD, shared_mem_SDD = (
        shared_batch_inputs["initialized_S"],
        shared_batch_inputs["shared_mem_S"],
        shared_batch_inputs["shared_mem_SD"],
        shared_batch_inputs["shared_mem_SDD"],
    )

    # Move to device and cast dtype
    x_SXXXD = shared_dirs_SXXXD.to(device).to(torch.float32)

    # Get inputs
    x_SXD = x_SXXXD.view(x_SXXXD.size(0), -1, x_SXXXD.size(4))
    counts_S = shared_mem_S.to(device)
    means_SD = shared_mem_SD.to(device)
    cmom2s_SDD = shared_mem_SDD.to(device)

    if not initialized_S[0]:
        # Compute initial second-moment stats
        means_SD = x_SXD.mean(dim=1)
        centered_SXD = x_SXD - means_SD[:, None, :]
        cmom2s_SDD = centered_SXD.transpose(1, 2) @ centered_SXD

        # Update counts
        counts_S = torch.ones(x_SXD.size(0), device=device) * x_SXD.size(1)
        initialized_S[0] = True

        del centered_SXD
    else:
        # Update counts
        counts_S += x_SXD.size(1)

        # Get existing tensors and apply Chan-style update
        # Update the mean according to the batch deviation from the old mean.
        deltas_SXD = x_SXD - means_SD[:, None, :]
        means_SD.add_(deltas_SXD.sum(dim=1) / counts_S[:, None])
        delta2s_SXD = x_SXD - means_SD[:, None, :]

        # Update the variance using the batch deviation
        for i in range(cmom2s_SDD.size(0)):
            cmom2s_SDD[i].addmm_(deltas_SXD[i].T, delta2s_SXD[i])

        del deltas_SXD, delta2s_SXD

    # Write results back to input rather than output
    shared_mem_S.copy_(counts_S)
    shared_mem_SD.copy_(means_SD)
    shared_mem_SDD.copy_(cmom2s_SDD)
    return {}


def handle_output_fn(
    shared_inputs: Dict[str, torch.Tensor],
    shared_outputs: Dict[str, torch.Tensor],
    batch_idx: int,
    covs: ParallelBatchCovariance,
):
    covs._counts_BN[batch_idx].copy_(shared_inputs["shared_mem_S"])
    covs._means_BND[batch_idx].copy_(shared_inputs["shared_mem_SD"])
    covs._cmom2s_BNDD[batch_idx].copy_(shared_inputs["shared_mem_SDD"])
    covs._initialized_B[batch_idx] = shared_inputs["initialized_S"][0]
