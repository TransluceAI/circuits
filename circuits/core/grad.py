"""
Utilities for gradient-based attribution.
"""

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
)


def _llama3_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension of estimator_X1D,
    then applying the same normalization to constant to X2D for all X1.

    We cast to float32 for numeric stability.
    """

    # Put everything on the device that input is on
    device = x_X1X2D.device

    # Compute
    return (
        norm_w_D[None, None, :].to(device)
        * x_X1X2D
        * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    )


def remove_forward_hooks(main_module: nn.Module):
    """Function to remove all forward and pre-forward hooks from a module and

    its sub-modules.
    """

    # Remove forward hooks
    for _, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks"):
            hooks = list(submodule._forward_hooks.keys())  # Get a list of hook IDs
            for hook_id in hooks:
                submodule._forward_hooks.pop(hook_id)

        # Remove pre-forward hooks
        if hasattr(submodule, "_forward_pre_hooks"):
            pre_hooks = list(submodule._forward_pre_hooks.keys())  # Get a list of pre-hook IDs
            for pre_hook_id in pre_hooks:
                submodule._forward_pre_hooks.pop(pre_hook_id)


class StopGradientModule(nn.Module):
    _stop_gradient = True


class StraightThroughLlamaRMSNorm(StopGradientModule):
    """
    Wrap an existing LlamaRMSNorm so that

      • forward  = real RMSNorm value
      • backward = identity wrt input  (dout/dx = I)
      • weight   is frozen (requires_grad = False)

    The underlying norm module stays exactly where it is
    (same device, dtype, optimizer state, etc.).
    """

    def __init__(self, norm: LlamaRMSNorm):
        super().__init__()
        self.norm = norm  # keep the SAME module
        self.norm.weight.requires_grad_(False)  # freeze its parameter
        self.weight = self.norm.weight  # ref ptr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        coeff = _llama3_layernorm_fn(
            x.new_ones(B * L, 1, 1),
            x.view(B * L, D),
            self.norm.weight,
            self.norm.variance_epsilon,
        ).detach()  # real RMSNorm
        return x * coeff.permute(1, 0, 2).view(B, L, D)


class ShapleySoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.softmax(x, dim=-1, dtype=torch.float32)
        ctx.save_for_backward(x, result)
        return result.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, result = ctx.saved_tensors
        with torch.no_grad():
            # R_l = grad_output.to(torch.float32) * result
            # R_prev = logits * (R_l - result * R_l.sum(-1, keepdim=True))
            # bias = (R_l - R_prev).sum(-1, keepdim=True) / (logits.shape[-1])
            # R_prev_eq_bias = R_prev + bias
            R_l = grad_output.to(torch.float32) * result
            total_R = R_l.sum(-1, keepdim=True)
            R_prev_eq_bias = total_R * result
            # print("OUTPUT FLOW", (R_l).sum().detach().item())
            # print("INPUT FLOW", (R_prev_eq_bias).sum().detach().item())
            # input()
            # exit(-1)
            grad_x = R_prev_eq_bias / logits
        return grad_x.to(logits.dtype), None


class ShapleyMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, y)
        return torch.matmul(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        with torch.no_grad():
            grad_x = torch.matmul(grad_output, y.transpose(-2, -1))
            grad_y = torch.matmul(x.transpose(-2, -1), grad_output)
            # out = (grad_output * torch.matmul(x, y)).sum()
            # inp = (grad_x * x * 0.5).sum() + (grad_y * y * 0.5).sum()
            # print("Matmul Shapley out:", float(out), "in:", float(inp))
        return grad_x * 0.5, grad_y * 0.5


def shapley_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    use_shapley_qk: bool = False,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # QK
    # attn_weights = ShapleyQK.apply(query, key_states, scaling, attention_mask, use_shapley_qk)
    if use_shapley_qk:
        attn_scores = ShapleyMatmul.apply(query, key_states.transpose(2, 3)) * scaling
    else:
        attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_scores = attn_scores + causal_mask

    # use final probs for attribution; keep in fp32 for stability
    if use_shapley_qk:
        attn_weights = ShapleySoftmax.apply(attn_scores).to(query.dtype)
    else:
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

    # OV
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if use_shapley_qk:
        attn_output = ShapleyMatmul.apply(
            attn_weights, value_states
        )  # gives 50% of the flow to each, QK does have grad
    else:
        attn_output = torch.matmul(
            attn_weights, value_states
        )  # gives 100% of the flow to both, and attn_weights has 0 grad so flow is maintained
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS["shapley"] = shapley_attention_forward


class NoQKGradAttention(StopGradientModule):
    """
    Wraps an existing LlamaAttention so that the soft‑maxed attention
    map gets no gradient.  The wrapped module's weights are untouched.
    """

    def __init__(self, attn: LlamaAttention, use_shapley_qk: bool = False):
        super().__init__()
        self.attn = attn  # keep the EXACT same module (weights, device, dtype)
        self.q_proj = attn.q_proj  # ref ptr
        self.k_proj = attn.k_proj  # ref ptr
        self.v_proj = attn.v_proj  # ref ptr
        self.o_proj = attn.o_proj  # ref ptr
        self.attn.config._attn_implementation = "shapley"
        self.use_shapley_qk = use_shapley_qk

    def forward(self, *args, **kwargs):
        kwargs["use_shapley_qk"] = self.use_shapley_qk
        attn_output, attn_weights = self.attn(*args, **kwargs)
        return attn_output, attn_weights


class StopGradGateMLP(StopGradientModule):
    """
    Wrap an existing LlamaMLP so the activation‑gate side
      act_fn( gate_proj(x) )
    is detached from the autograd graph.
    """

    def __init__(self, mlp: LlamaMLP):
        super().__init__()
        self.mlp = mlp
        for p in self.mlp.gate_proj.parameters():
            p.requires_grad_(False)
        self.down_proj = self.mlp.down_proj  # ref ptr
        self.act_fn = self.mlp.act_fn  # ref ptr
        self.gate_proj = self.mlp.gate_proj  # ref ptr
        self.up_proj = self.mlp.up_proj  # ref ptr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_act = self.mlp.act_fn(self.mlp.gate_proj(x)).detach()
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(gate_act * up_branch)


class StopGradMLP(StopGradientModule):
    """
    Wrap an existing LlamaMLP and stop *all* gradient through it.
    """

    def __init__(self, mlp: LlamaMLP):
        super().__init__()
        self.mlp = mlp
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        self.down_proj = self.mlp.down_proj  # ref ptr
        self.act_fn = self.mlp.act_fn  # ref ptr
        self.gate_proj = self.mlp.gate_proj  # ref ptr
        self.up_proj = self.mlp.up_proj  # ref ptr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.mlp(x)
        return out.detach()


silu_fn = torch.nn.SiLU()


class ShapleySiLUGate(torch.autograd.Function):
    """
    Approximate Shapley gradient for linear + SiLU gate. Distributes the attribution using closed-form
    analytical approximation for linear + ReLU along with an empirical correction for the difference
    between ReLU and SiLU.
    Reference for linear + ReLU: https://arxiv.org/abs/1909.06143
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        """
        Compute y = SiLU(x @ W^T + b) and save what's needed for a faithful backward.
        Shapes:
          x:      (..., D_in)
          weight: (D_out, D_in)
          bias:   (D_out,) or None
        """
        z = x.matmul(weight.transpose(-1, -2))
        if bias is not None:
            z = z + bias

        y = silu_fn(z)

        # Save for backward
        ctx.save_for_backward(x, weight, bias, z)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias, _ = ctx.saved_tensors
        eps = 1e-12

        with torch.no_grad():
            # print("SHAPLEY SILU GATE")
            # print("output flow", (grad_output * silu_fn(x.matmul(weight.transpose(-1, -2)))).sum().detach().item())
            W = weight  # (D_ffn, D_model)
            WT = W.transpose(-1, -2)  # (D_model, D_ffn)

            # --- stats without per_output_dim ---
            total = x.matmul(WT)  # (..., D_ffn)      = sum_m x_m * W_fm
            total_sq = x.square().matmul(W.square().transpose(-1, -2))  # (..., D_ffn)

            mu = 0.5 * total
            var = total_sq / 6.0 + total.square() / 12.0
            std = var.clamp_min(eps).sqrt()

            # Faster than torch.distributions.Normal(...).cdf(...)
            # (also casts to f32 for numeric stability if needed)
            ratio = mu / std
            if ratio.dtype in (torch.float16, torch.bfloat16):
                cdf = torch.special.ndtr(ratio.to(torch.float32)).to(ratio.dtype)
            else:
                cdf = torch.special.ndtr(ratio)

            post_silu = silu_fn(total)  # SiLU on the bias-free total

            # --- first term: sum_f x_m * W_fm * (cdf_f * grad_f)  ->  (x *)[(cdf*grad) @ W]
            u = cdf * grad_output  # (..., D_ffn)
            term1 = u.matmul(W)  # (..., D_model)
            term1 = term1  # will be used as-is after safe divide

            # --- residual redistribution (proportional to |x_m|*|W_fm|; cdf cancels) ---
            denom = x.abs().matmul(W.abs().transpose(-1, -2)).clamp_min(eps)  # (..., D_ffn)
            resid = post_silu - cdf * total  # (..., D_ffn)

            T = (resid * grad_output) / denom  # (..., D_ffn)
            term2 = x.sign() * T.matmul(W.abs())  # (..., D_model)

            # Combine and apply your "SAFE divide" semantics:
            # originally: grad_x_attr = (x * term1 + |x| * term2_base) / x, 0 if |x|<eps
            # which simplifies to: term1 + sign(x) * term2_base, masked where |x|<eps
            grad_x_attr = term1 + term2
            grad_x_attr = torch.where(x.abs() >= eps, grad_x_attr, torch.zeros_like(grad_x_attr))
            # print("input flow", (grad_x_attr * x).sum().detach().item())

        grad_x = grad_x_attr
        # No grads for weights/bias in this branch
        grad_w = torch.zeros_like(weight)
        grad_b = torch.zeros_like(bias) if bias is not None else None
        return grad_x, grad_w, grad_b


class ShapleyElementwiseMult(torch.autograd.Function):
    """
    Shapley gradient for elementwise multiplication. This distributes the attribution equally to
    both branches, avoiding double-counting (which normal gradient would do).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, use_half_rule: bool = True):
        ctx.save_for_backward(x, y)
        ctx.use_half_rule = use_half_rule
        return x * y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        return (
            (0.5 if ctx.use_half_rule else 1.0) * grad_output * y,
            (0.5 if ctx.use_half_rule else 1.0) * grad_output * x,
            None,
        )


class ShapleyGradMLP(StopGradientModule):
    """
    Shapley gradient for an entire gated MLP.
    """

    def __init__(self, mlp: LlamaMLP, use_half_rule: bool = True):
        super().__init__()
        self.mlp = mlp
        self.gate_proj = self.mlp.gate_proj  # ref ptr
        self.up_proj = self.mlp.up_proj  # ref ptr
        self.down_proj = self.mlp.down_proj  # ref ptr
        self.use_half_rule = use_half_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.mlp.gate_proj.weight
        b = self.mlp.gate_proj.bias

        gate_act = ShapleySiLUGate.apply(x, W, b)
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(
            ShapleyElementwiseMult.apply(gate_act, up_branch, self.use_half_rule)
        )


class RelPGradMLP(StopGradientModule):
    """
    Shapley gradient for an entire gated MLP.
    """

    def __init__(self, mlp: LlamaMLP, use_half_rule: bool = True):
        super().__init__()
        self.mlp = mlp
        self.down_proj = self.mlp.down_proj  # ref ptr
        self.gate_proj = self.mlp.gate_proj  # ref ptr
        self.up_proj = self.mlp.up_proj  # ref ptr
        self.use_half_rule = use_half_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = torch.sigmoid(
            self.mlp.gate_proj(x)
        ).detach()  # treat as constant, so grads just get multiplied by it
        gate_proj = self.mlp.gate_proj(x)
        gate_act = gate_proj * coeff
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(
            ShapleyElementwiseMult.apply(gate_act, up_branch, self.use_half_rule)
        )


def stop_nonlinear_grad_for_llama(
    model,
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    use_half_rule: bool = True,
):
    """
    Stop gradient for all non-linear layers in the model.
    """
    # replacement model with stop grad on non-linear ops
    model.model.norm = StraightThroughLlamaRMSNorm(model.model.norm)
    for layer in range(len(model.model.layers)):
        # all layernorms, requiring weight copying
        model.model.layers[layer].input_layernorm = StraightThroughLlamaRMSNorm(
            model.model.layers[layer].input_layernorm
        )
        model.model.layers[layer].post_attention_layernorm = StraightThroughLlamaRMSNorm(
            model.model.layers[layer].post_attention_layernorm
        )
        # attention weights are not needed for gradient computation
        model.model.layers[layer].self_attn = NoQKGradAttention(
            model.model.layers[layer].self_attn, use_shapley_qk
        )
        # mlp gate is not needed for gradient computation
        if use_relp_grad:
            model.model.layers[layer].mlp = RelPGradMLP(
                model.model.layers[layer].mlp, use_half_rule
            )
        elif use_shapley_grad:
            model.model.layers[layer].mlp = ShapleyGradMLP(
                model.model.layers[layer].mlp, use_half_rule
            )
        else:
            model.model.layers[layer].mlp = StopGradGateMLP(model.model.layers[layer].mlp)
    return model


def revert_stop_nonlinear_grad_for_llama(model):
    """
    Revert stop gradient for all non-linear layers in the model.
    """
    model.model.norm = model.model.norm.norm
    for layer in range(len(model.model.layers)):
        # all layernorms, requiring weight copying
        model.model.layers[layer].input_layernorm = model.model.layers[layer].input_layernorm.norm
        model.model.layers[layer].post_attention_layernorm = model.model.layers[
            layer
        ].post_attention_layernorm.norm
        # attention weights are not needed for gradient computation
        model.model.layers[layer].self_attn = model.model.layers[layer].self_attn.attn
        # mlp gate is not needed for gradient computation
        model.model.layers[layer].mlp = model.model.layers[layer].mlp.mlp
    return model


def layerwise_stop_nonlinear_grad_for_llama(
    model,
    start_layer: int,
    end_layer: int,
    use_shapley_grad: bool = False,
    use_shapley_qk: bool = False,
    use_relp_grad: bool = False,
    use_stop_grad_on_mlps: bool = True,
    use_half_rule: bool = True,
):
    # replacement model with stop grad on non-linear ops
    model.model.norm = StraightThroughLlamaRMSNorm(model.model.norm)
    # for the start and the end layer, we don't do stop grad on mlp
    for layer in [start_layer, end_layer]:
        if layer < 0 or layer >= len(model.model.layers):
            continue  # no-op if layer is out of range
        # all layernorms, requiring weight copying
        model.model.layers[layer].input_layernorm = StraightThroughLlamaRMSNorm(
            model.model.layers[layer].input_layernorm
        )
        model.model.layers[layer].post_attention_layernorm = StraightThroughLlamaRMSNorm(
            model.model.layers[layer].post_attention_layernorm
        )
        # attention
        model.model.layers[layer].self_attn = NoQKGradAttention(
            model.model.layers[layer].self_attn, use_shapley_qk
        )
        # mlp
        if use_relp_grad:
            model.model.layers[layer].mlp = RelPGradMLP(
                model.model.layers[layer].mlp, use_half_rule
            )
        elif use_shapley_grad:
            model.model.layers[layer].mlp = ShapleyGradMLP(
                model.model.layers[layer].mlp, use_half_rule
            )
        else:
            model.model.layers[layer].mlp = StopGradGateMLP(model.model.layers[layer].mlp)

    # for layers in between, we do stop grad on mlp
    for layer in range(start_layer + 1, end_layer):
        if layer < 0 or layer >= len(model.model.layers):
            continue  # no-op if layer is out of range
        # all layernorms, requiring weight copying
        model.model.layers[layer].input_layernorm = StraightThroughLlamaRMSNorm(
            model.model.layers[layer].input_layernorm
        )
        model.model.layers[layer].post_attention_layernorm = StraightThroughLlamaRMSNorm(
            model.model.layers[layer].post_attention_layernorm
        )
        # attention
        model.model.layers[layer].self_attn = NoQKGradAttention(
            model.model.layers[layer].self_attn, use_shapley_qk
        )
        # mlp
        model.model.layers[layer].mlp = StopGradMLP(model.model.layers[layer].mlp)

    return model


def layerwise_revert_stop_nonlinear_grad_for_llama(
    model,
    start_layer: int,
    end_layer: int,
):
    model.model.norm = model.model.norm.norm
    for layer in range(start_layer, end_layer + 1):
        if layer < 0 or layer >= len(model.model.layers):
            continue  # no-op if layer is out of range
        # all layernorms, requiring weight copying
        model.model.layers[layer].input_layernorm = model.model.layers[layer].input_layernorm.norm
        model.model.layers[layer].post_attention_layernorm = model.model.layers[
            layer
        ].post_attention_layernorm.norm
        # attention weights are not needed for gradient computation
        model.model.layers[layer].self_attn = model.model.layers[layer].self_attn.attn
        # mlp gate is not needed for gradient computation
        model.model.layers[layer].mlp = model.model.layers[layer].mlp.mlp
    return model
