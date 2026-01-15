from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class LayerActivations(BaseModel):
    resid_BTD: Tensor | None = None
    mlp_in_BTD: Tensor | None = None
    mlp_gate_BTD: Tensor | None = None
    mlp_out_BTD: Tensor | None = None
    attn_heads_BTD: Tensor | None = None
    attn_out_BTD: Tensor | None = None
    attn_map_BQTT: Tensor | None = None
    neurons_BTI: Tensor | None = None
    # for gemma2-like models
    pre_mlp_norms_in_BTD: Tensor | None = None
    post_mlp_norms_in_BTD: Tensor | None = None
    post_attention_norms_in_BTD: Tensor | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to(
        self, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> LayerActivations:
        return LayerActivations(
            resid_BTD=(
                self.resid_BTD.to(device=device, dtype=dtype)
                if self.resid_BTD is not None
                else None
            ),
            mlp_in_BTD=(
                self.mlp_in_BTD.to(device=device, dtype=dtype)
                if self.mlp_in_BTD is not None
                else None
            ),
            mlp_gate_BTD=(
                self.mlp_gate_BTD.to(device=device, dtype=dtype)
                if self.mlp_gate_BTD is not None
                else None
            ),
            mlp_out_BTD=(
                self.mlp_out_BTD.to(device=device, dtype=dtype)
                if self.mlp_out_BTD is not None
                else None
            ),
            attn_heads_BTD=(
                self.attn_heads_BTD.to(device=device, dtype=dtype)
                if self.attn_heads_BTD is not None
                else None
            ),
            attn_out_BTD=(
                self.attn_out_BTD.to(device=device, dtype=dtype)
                if self.attn_out_BTD is not None
                else None
            ),
            attn_map_BQTT=(
                self.attn_map_BQTT.to(device=device, dtype=dtype)
                if self.attn_map_BQTT is not None
                else None
            ),
            neurons_BTI=(
                self.neurons_BTI.to(device=device, dtype=dtype)
                if self.neurons_BTI is not None
                else None
            ),
            pre_mlp_norms_in_BTD=(
                self.pre_mlp_norms_in_BTD.to(device=device, dtype=dtype)
                if self.pre_mlp_norms_in_BTD is not None
                else None
            ),
            post_mlp_norms_in_BTD=(
                self.post_mlp_norms_in_BTD.to(device=device, dtype=dtype)
                if self.post_mlp_norms_in_BTD is not None
                else None
            ),
            post_attention_norms_in_BTD=(
                self.post_attention_norms_in_BTD.to(device=device, dtype=dtype)
                if self.post_attention_norms_in_BTD is not None
                else None
            ),
        )

    def share_memory_(self):
        if self.resid_BTD is not None:
            self.resid_BTD.share_memory_()
        if self.mlp_in_BTD is not None:
            self.mlp_in_BTD.share_memory_()
        if self.mlp_gate_BTD is not None:
            self.mlp_gate_BTD.share_memory_()
        if self.mlp_out_BTD is not None:
            self.mlp_out_BTD.share_memory_()
        if self.attn_heads_BTD is not None:
            self.attn_heads_BTD.share_memory_()
        if self.attn_out_BTD is not None:
            self.attn_out_BTD.share_memory_()
        if self.attn_map_BQTT is not None:
            self.attn_map_BQTT.share_memory_()
        if self.neurons_BTI is not None:
            self.neurons_BTI.share_memory_()
        if self.pre_mlp_norms_in_BTD is not None:
            self.pre_mlp_norms_in_BTD.share_memory_()
        if self.post_mlp_norms_in_BTD is not None:
            self.post_mlp_norms_in_BTD.share_memory_()
        if self.post_attention_norms_in_BTD is not None:
            self.post_attention_norms_in_BTD.share_memory_()
        return self

    def slice_batch(self, slice: slice) -> LayerActivations:
        return LayerActivations(
            resid_BTD=self.resid_BTD[slice] if self.resid_BTD is not None else None,
            mlp_in_BTD=self.mlp_in_BTD[slice] if self.mlp_in_BTD is not None else None,
            mlp_gate_BTD=self.mlp_gate_BTD[slice] if self.mlp_gate_BTD is not None else None,
            mlp_out_BTD=self.mlp_out_BTD[slice] if self.mlp_out_BTD is not None else None,
            attn_heads_BTD=self.attn_heads_BTD[slice] if self.attn_heads_BTD is not None else None,
            attn_out_BTD=self.attn_out_BTD[slice] if self.attn_out_BTD is not None else None,
            attn_map_BQTT=self.attn_map_BQTT[slice] if self.attn_map_BQTT is not None else None,
            neurons_BTI=self.neurons_BTI[slice] if self.neurons_BTI is not None else None,
            pre_mlp_norms_in_BTD=(
                self.pre_mlp_norms_in_BTD[slice] if self.pre_mlp_norms_in_BTD is not None else None
            ),
            post_mlp_norms_in_BTD=(
                self.post_mlp_norms_in_BTD[slice]
                if self.post_mlp_norms_in_BTD is not None
                else None
            ),
            post_attention_norms_in_BTD=(
                self.post_attention_norms_in_BTD[slice]
                if self.post_attention_norms_in_BTD is not None
                else None
            ),
        )

    def mem_usage_gb(self):
        ans = 0
        for v in self.model_dump().values():
            if isinstance(v, Tensor):
                ans += v.nbytes
        return ans / 1e9


class ModelActivations(BaseModel):
    layers: dict[int, LayerActivations]
    unembed_in_BTD: Tensor | None = None
    unembed_out_BTV: Tensor | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getitem__(self, item: int) -> LayerActivations:
        return self.layers[item]

    def slice_batch(self, slice: slice) -> ModelActivations:
        return ModelActivations(
            layers={k: v.slice_batch(slice) for k, v in self.layers.items()},
            unembed_in_BTD=self.unembed_in_BTD[slice] if self.unembed_in_BTD is not None else None,
            unembed_out_BTV=(
                self.unembed_out_BTV[slice] if self.unembed_out_BTV is not None else None
            ),
        )

    def share_memory_(self):
        if self.unembed_in_BTD is not None:
            self.unembed_in_BTD.share_memory_()
        if self.unembed_out_BTV is not None:
            self.unembed_out_BTV.share_memory_()
        for layer_acts in self.layers.values():
            layer_acts.share_memory_()
        return self

    def to(
        self, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> ModelActivations:
        return ModelActivations(
            layers={k: v.to(device=device, dtype=dtype) for k, v in self.layers.items()},
            unembed_in_BTD=(
                self.unembed_in_BTD.to(device=device, dtype=dtype)
                if self.unembed_in_BTD is not None
                else None
            ),
            unembed_out_BTV=(
                self.unembed_out_BTV.to(device=device, dtype=dtype)
                if self.unembed_out_BTV is not None
                else None
            ),
        )

    def mem_usage_gb(self):
        ans = 0
        for v in self.model_dump().values():
            if isinstance(v, Tensor):
                ans += v.nbytes
        ans /= 1e9
        for layer_dict in self.layers.values():
            ans += layer_dict.mem_usage_gb()
        return ans
