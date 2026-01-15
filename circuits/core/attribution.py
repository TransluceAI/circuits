"""
Utility classes and functions for attributing CLSO effects between model components.
"""

import torch
from opt_einsum import contract
from pydantic import BaseModel, ConfigDict
from util.subject import Subject


class CLSOBuffer(BaseModel):
    layer: int
    token: int
    final_attribution_BNsNf: torch.Tensor
    frozen_final_attribution_BNsNf: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    take_diff: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def apply_mask(self, indices: torch.Tensor):
        pass

    def get_activations(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def shape(self) -> torch.Size:
        raise NotImplementedError("Subclasses must implement this method")

    def decode(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class LogitsBuffer(CLSOBuffer):
    ln_B1D: torch.Tensor
    unembed_VD: torch.Tensor
    logits_BV: torch.Tensor
    keep_indices: set[int] = set()
    final_logit_softcapping_BV: torch.Tensor | None = None  # only for gemma2
    neuron_indices_map: dict[int, int] = {}

    def get_activations(self) -> torch.Tensor:
        return (
            self.logits_BV
            if not self.take_diff
            else self.logits_BV - self.logits_BV.mean(dim=0, keepdim=True)
        )

    @property
    def shape(self) -> torch.Size:
        return self.logits_BV.shape

    def apply_softcapping(self, logits_contrib_BNV: torch.Tensor) -> torch.Tensor:
        if self.final_logit_softcapping_BV is not None:
            logits_contrib_BNV = logits_contrib_BNV * self.final_logit_softcapping_BV.unsqueeze(
                dim=1
            )
        return logits_contrib_BNV

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return contract("bnd,vd->bnv", x * self.ln_B1D, self.unembed_VD)


class MLPBuffer(CLSOBuffer):
    ln_in_B1D: torch.Tensor
    ln_out_B1D: torch.Tensor | None = None  # only for gemma2
    mlp_gate_BN: torch.Tensor
    w_in_ND: torch.Tensor
    w_out_DN: torch.Tensor
    neurons_BN: torch.Tensor
    neuron_indices_map: dict[int, int] = {}
    keep_indices: set[int] = set()

    def apply_mask(self, indices: torch.Tensor):
        self.indices = indices
        self.mlp_gate_BN = self.mlp_gate_BN[:, indices].contiguous()
        self.w_in_ND = self.w_in_ND[indices, :].contiguous()
        self.w_out_DN = self.w_out_DN[:, indices].contiguous()
        self.neurons_BN = self.neurons_BN[:, indices].contiguous()
        self.final_attribution_BNsNf = self.final_attribution_BNsNf[:, indices, :].contiguous()

        # reindex
        if bool(self.neuron_indices_map):
            new_neuron_indices_map = {
                i: self.neuron_indices_map[idx] for i, idx in enumerate(sorted(indices.tolist()))
            }
            self.neuron_indices_map = new_neuron_indices_map
        else:
            self.neuron_indices_map = {i: idx for i, idx in enumerate(sorted(indices.tolist()))}

    def get_activations(self) -> torch.Tensor:
        return (
            self.neurons_BN
            if not self.take_diff
            else self.neurons_BN - self.neurons_BN.mean(dim=0, keepdim=True)
        )

    def get_activations_without_bias(self) -> torch.Tensor:
        return self.get_activations()

    @property
    def shape(self) -> torch.Size:
        return self.neurons_BN.shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # apply layernorm and w_in
        result = (
            torch.matmul(
                x * self.ln_in_B1D, self.w_in_ND.T
            )  # [B, N_in, D] x [D, N] -> [B, N_in, N]
            * self.mlp_gate_BN[:, None, :]
        )
        return result  # [B, N_in, N]

    def decode(self) -> torch.Tensor:
        # apply w_out
        result = contract("bn,dn->bnd", self.get_activations(), self.w_out_DN)
        if self.ln_out_B1D is not None:
            result = result * self.ln_out_B1D
        return result


class EmbedBuffer(CLSOBuffer):
    embed_B1D: torch.Tensor
    keep_indices: set[int] = set()
    neuron_indices_map: dict[int, int] = {0: 0}

    def get_activations(self) -> torch.Tensor:
        return (
            torch.ones(self.embed_B1D.size(0), self.embed_B1D.size(1), device=self.embed_B1D.device)
            if not self.take_diff
            else torch.zeros(
                self.embed_B1D.size(0), self.embed_B1D.size(1), device=self.embed_B1D.device
            )
        )

    def decode(self) -> torch.Tensor:
        return self.embed_B1D

    @property
    def shape(self) -> torch.Size:
        return self.embed_B1D.shape[:-1]


class TranscoderBuffer(CLSOBuffer):
    ln_B1D: torch.Tensor
    # transcoder: JumpReluAutoEncoder
    W_enc: torch.Tensor  # shape: [D, N]
    W_dec: torch.Tensor  # shape: [N, D]
    b_enc: torch.Tensor  # shape: [N]
    b_dec: torch.Tensor  # shape: [D]
    threshold: torch.Tensor  # shape: [N]
    apply_b_dec_to_input: bool
    neurons_BN: torch.Tensor
    keep_indices: set[int] = set()

    def get_activations(self) -> torch.Tensor:
        return (
            self.neurons_BN
            if not self.take_diff
            else self.neurons_BN - self.neurons_BN.mean(dim=0, keepdim=True)
        )

    def get_activations_without_bias(self) -> torch.Tensor:
        if self.take_diff:
            return (self.neurons_BN - self.neurons_BN.mean(dim=0, keepdim=True)) - (
                self.b_enc * (self.neurons_BN > 0)
            )
        else:
            return self.neurons_BN - (self.b_enc * (self.neurons_BN > 0))

    @property
    def shape(self) -> torch.Size:
        return self.neurons_BN.shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, N_in, D]
        result = x * self.ln_B1D
        # if self.apply_b_dec_to_input:
        #     result = result - self.b_dec[None, None, :]
        result = torch.matmul(result, self.W_enc)  # [B, N_in, D] x [D, N] -> [B, N_in, N]
        result = result * (self.neurons_BN > 0)[:, None, :]
        # x = torch.nn.functional.relu(x * (x > self.threshold))
        return result

    def decode(self) -> torch.Tensor:
        # print(
        #     f"nonzero neurons, layer {self.layer}, token {self.token}: {self.neurons_BN[0].nonzero().flatten().shape} / {self.neurons_BN[0].numel()}"
        # )
        result = contract("bn,nd->bnd", self.neurons_BN, self.W_dec)
        # result += self.transcoder.b_dec
        return result

    def apply_mask(self, indices: torch.Tensor):
        self.indices = indices
        self.W_enc = self.W_enc[:, indices].contiguous()
        self.W_dec = self.W_dec[indices, :].contiguous()
        self.b_enc = self.b_enc[indices].contiguous()
        self.threshold = self.threshold[indices].contiguous()
        # self.transcoder.b_dec.data should not be modified
        self.neurons_BN = self.neurons_BN[:, indices].contiguous()
        self.final_attribution_BNsNf = self.final_attribution_BNsNf[:, indices, :].contiguous()


def attribute(
    subject: Subject,
    ov_sum_B1TsDD: torch.Tensor | None,
    start_buffer: CLSOBuffer,
    finish_buffer: CLSOBuffer,
    keep_fo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # we need to compute the total contribution via attns + FO
    contrib_BNsNf = torch.zeros(
        (
            start_buffer.shape[0],
            start_buffer.shape[1],
            finish_buffer.shape[1],
        ),
        device=start_buffer.get_activations().device,
    )
    contrib_fo_BNsNf = torch.zeros_like(contrib_BNsNf)
    has_so = ov_sum_B1TsDD is not None
    has_fo = (start_buffer.token == finish_buffer.token) and keep_fo

    if has_so or has_fo:
        # get output of MLP
        dirs_BND = start_buffer.decode()

    # SO effect
    if has_so:
        # get contribution of MLP to logits via all attns
        ov_local_BDD = ov_sum_B1TsDD[:, 0, start_buffer.token, :, :]
        contrib_BND = torch.bmm(dirs_BND, ov_local_BDD.permute(0, 2, 1))
        contrib_BNsNf = finish_buffer.encode(contrib_BND)

    # FO effect
    if has_fo:
        contrib_fo_BNsNf = finish_buffer.encode(dirs_BND)

    # also compute relative attribution with adaptive epsilon for numerical stability
    target_activations = finish_buffer.get_activations()[:, None, :]
    eps = target_activations.abs().mean() * 1e-6  # adaptive epsilon based on activation magnitude
    relative_BNsNf = contrib_BNsNf / (target_activations + eps)
    relative_fo_BNsNf = contrib_fo_BNsNf / (target_activations + eps)

    return contrib_BNsNf, relative_BNsNf, contrib_fo_BNsNf, relative_fo_BNsNf


def attribute_lightweight(
    subject: Subject,
    ov_sum_B1TsDD: torch.Tensor | None,
    start_buffer: CLSOBuffer,
    finish_buffer: CLSOBuffer,
    keep_fo: bool = True,
    return_absolute: bool = False,
) -> torch.Tensor:
    contrib_BNsNf, relative_BNsNf, contrib_fo_BNsNf, relative_fo_BNsNf = attribute(
        subject, ov_sum_B1TsDD, start_buffer, finish_buffer, keep_fo
    )
    if return_absolute:
        return (contrib_BNsNf + contrib_fo_BNsNf) if keep_fo else contrib_BNsNf
    return (relative_BNsNf + relative_fo_BNsNf) if keep_fo else relative_BNsNf
