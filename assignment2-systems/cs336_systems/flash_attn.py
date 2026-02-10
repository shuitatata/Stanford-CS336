import torch
from torch import Tensor
from jaxtyping import Float
from math import ceil
from torch.autograd.function import FunctionCtx
import triton
from triton import language as tl


class PytorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        is_causal=False,
    ):
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        d_k = Q.shape[-1]
        d_v = V.shape[-1]

        T_q = ceil(N_q / Q_TILE_SIZE)
        T_k = ceil(N_k / K_TILE_SIZE)

        O = torch.zeros(Q.shape[:-1] + (d_v,), device=Q.device)
        L = torch.zeros(Q.shape[:-1], device=Q.device)

        for i in range(T_q):
            Q_tile = Q[..., i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE, :]
            l = torch.zeros(Q_tile.shape[:-1], device=Q.device)  # [B, T_q_tile]
            m = torch.zeros(Q_tile.shape[:-1], device=Q.device) - float(
                "inf"
            )  # [B, T_q_tile]
            O_tile = torch.zeros(Q_tile.shape[:-1] + (d_v,), device=Q.device)

            for j in range(T_k):
                K_tile = K[..., j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE, :]
                V_tile = V[..., j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE, :]

                attn_score = torch.matmul(Q_tile, K_tile.transpose(-1, -2)) / (
                    d_k**0.5
                )  # [B, T_q_tile, T_k_tile]
                new_m = torch.max(m, attn_score.max(dim=-1).values)  # [B, T_q_tile]

                p = torch.exp(
                    attn_score - new_m.unsqueeze(-1)
                )  # [B, T_q_tile, T_k_tile]
                alpha = torch.exp(m - new_m)
                l = alpha * l + p.sum(dim=-1)  # [B, T_q_tile]
                O_tile = alpha.unsqueeze(-1) * O_tile + torch.matmul(
                    p, V_tile
                )  # [B, T_q_tile, d_v]
                m = new_m

            # Write back the tile to the output
            O[..., i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE, :] = O_tile / l.unsqueeze(
                -1
            )
            L[..., i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        dO: Float[Tensor, "... queries d_v"],
    ):
        L, Q, K, V, O = ctx.saved_tensors

        d_k = Q.shape[-1]

        S = Q @ K.transpose(-1, -2) / (d_k**0.5)  # [B, T_q, T_k]
        P = torch.exp(S - L[..., :, None])  # [B, T_q, T_k]
        dV = P.transpose(-1, -2) @ dO  # [B, T_k, d_v]
        dP = dO @ V.transpose(-1, -2)  # [B, T_q, T_k]

        D = torch.sum(O * dO, dim=-1, keepdim=True)  # [B, T_q, 1]
        dS = P * (dP - D)  # [B, T_q, T_k]
        dQ = dS @ K / (d_k**0.5)  # [B, T_q, d_k]
        dK = dS.transpose(-1, -2) @ Q / (d_k**0.5)  # [B, T_k, d_k]

        return dQ, dK, dV, None


@triton.jit
# fmt: off
def flash_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr=False,
):
# fmt: on
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Initialize block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),   
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Initialize tile accumulators
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_tile = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    n_tile = tl.cdiv(N_KEYS, K_TILE_SIZE) # number of tiles along the key dimension

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)  # [T_q, D]
    for key_tile_index in range(n_tile):
        K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)  # [D, T_k]
        V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")  # [T_k, D]
        
        attn_scores = tl.dot(Q_tile, K_tile) * scale  # [T_q, T_k]

        # apply causal mask
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:, None] < k_indices[None, :]
            attn_scores = tl.where(mask, -1e6, attn_scores)

        new_m = tl.maximum(m_tile, tl.max(attn_scores, axis=1))  # [T_q]
        P = tl.exp(attn_scores - new_m[:, None])  # [T_q, T_k]

        # online softmax
        alpha = tl.exp(m_tile - new_m)  # [T_q]
        l_tile = alpha * l_tile + tl.sum(P, axis=1)  # [T_q]
        O_tile = alpha[:, None] * O_tile
        P = P.to(V_tile.dtype)
        O_tile = tl.dot(P, V_tile, acc=O_tile)

        # update
        m_tile = new_m
        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(O_block_ptr, (O_tile / l_tile[:, None]).to(O_block_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(L_block_ptr, (m_tile + tl.log(l_tile)).to(L_block_ptr.type.element_ty), boundary_check=(0,))


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        is_causal=False,
    ):
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        d_k = Q.shape[-1]
        d_v = V.shape[-1]

        T_q = ceil(N_q / Q_TILE_SIZE)

        O = torch.zeros(Q.shape[:-1] + (d_v,), device=Q.device, dtype=V.dtype)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)

        scale = 1 / (d_k**0.5)

        grid = (T_q, Q.shape[0])  # (num_query_tiles, batch_size)

        # fmt: off
        flash_forward_kernel[grid](
            Q, K, V, O, L,
            Q.stride(-3), Q.stride(-2), Q.stride(-1),
            K.stride(-3), K.stride(-2), K.stride(-1),
            V.stride(-3), V.stride(-2), V.stride(-1),
            O.stride(-3), O.stride(-2), O.stride(-1),
            L.stride(-2), L.stride(-1),
            N_q, N_k, scale,
            d_k,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )
        # fmt: on

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        dO: Float[Tensor, "... queries d_v"],
    ):

        raise NotImplementedError()
