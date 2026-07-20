from __future__ import annotations

import types


def apply_rotary_enc_real(xq, xk, cosine, sine, repeat_freqs_k=False):
    """Apply SAM2 axial RoPE without complex tensors.

    ``cosine`` and ``sine`` have shape ``[tokens, head_dim / 2]``. This is
    algebraically identical to multiplying adjacent feature pairs by the
    complex tensor returned by SAM2's ``compute_axial_cis``.
    """

    import torch

    def rotate(x, cos, sin):
        even = x.float()[..., 0::2]
        odd = x.float()[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).type_as(x)

    q_cos = cosine.view(1, 1, cosine.shape[0], cosine.shape[1])
    q_sin = sine.view(1, 1, sine.shape[0], sine.shape[1])
    query = rotate(xq, q_cos, q_sin)
    if xk.shape[-2] == 0:
        return query, xk
    if repeat_freqs_k:
        repetitions = xk.shape[-2] // xq.shape[-2]
        q_cos = q_cos.unsqueeze(2).expand(-1, -1, repetitions, -1, -1).flatten(2, 3)
        q_sin = q_sin.unsqueeze(2).expand(-1, -1, repetitions, -1, -1).flatten(2, 3)
    return query, rotate(xk, q_cos, q_sin)


def _real_rope_forward(self, q, k, v, num_k_exclude_rope=0):
    import torch
    from torch.nn import functional as functional

    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    num_k_rope = k.shape[-2] - num_k_exclude_rope
    rotated_q, rotated_k = apply_rotary_enc_real(
        q,
        k[:, :, :num_k_rope],
        self.rope_cosine,
        self.rope_sine,
        repeat_freqs_k=self.rope_k_repeat,
    )
    k = torch.cat((rotated_k, k[:, :, num_k_rope:]), dim=-2)
    dropout = self.dropout_p if self.training else 0.0
    output = functional.scaled_dot_product_attention(
        rotated_q, k, v, dropout_p=dropout
    )
    return self.out_proj(self._recombine_heads(output))


def patch_real_rope(model) -> int:
    """Replace every SAM2 RoPEAttention forward with an ONNX-safe equivalent."""
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "RoPEAttention":
            continue
        frequencies = module.freqs_cis.detach()
        module.register_buffer(
            "rope_cosine", frequencies.real.float().contiguous(), persistent=True
        )
        module.register_buffer(
            "rope_sine", frequencies.imag.float().contiguous(), persistent=True
        )
        delattr(module, "freqs_cis")
        module.forward = types.MethodType(_real_rope_forward, module)
        patched += 1
    if patched == 0:
        raise RuntimeError("model contains no RoPEAttention modules")
    return patched
