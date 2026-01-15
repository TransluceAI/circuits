import torch


def malloc_cpu(shape: tuple[int, ...], dtype: torch.dtype, shared: bool = False):
    x = torch.zeros(shape, dtype=dtype, device="cpu")
    if shared:
        x.share_memory_()
    return x


def malloc_cpu_if_None(
    x: torch.Tensor | None,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    shared: bool = False,
    verbose: bool = False,
):
    # Use original tensor if it's already allocated
    if x is not None:
        assert x.shape == shape, f"Expected shape {shape} but got {x.shape}"
        assert x.dtype == dtype, f"Expected dtype {dtype} but got {x.dtype}"
        assert x.device == torch.device("cpu"), f"Expected device 'cpu' but got {x.device}"
        if shared:
            assert x.is_shared()
        return x

    # Otherwise, allocate new tensor
    else:
        if verbose:
            print(
                f"Mallocing {shape=}, {dtype=}, {shared=} on CPU... ",
                end="",
                flush=True,
            )

        x = torch.zeros(shape, dtype=dtype, device="cpu")
        if shared:
            x.share_memory_()

        if verbose:
            print("Done")
        return x
