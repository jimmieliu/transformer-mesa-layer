from common import *


class Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init="final")

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_lma:
                Whether to use low-memory attention
            q_chunk_size:
                Query chunk size (for LMA)
            kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        # [*, Q/K, H, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        o = _mesa_attention(q, k, v, biases)

        o = self._wrap_up(o, q_x)

        return o


# IMPLEMENT LATER
# class MESA_ATTN(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_):
#         return _mesa_attention_fwd(input_)
#     @staticmethod
#     def backward(ctx, grad_output):
#         return _mesa_attention_bwd


def _mesa_attention(
    query: torch.Tensor,  # [*, L, H, C_hidden]
    key: torch.Tensor,  # [*, L, H, C_hidden]
    value: torch.Tensor,  # [*, L, H, C_hidden]
):  # biases: List[torch.Tensor],
    # [*, H, C_hidden, C_hidden]
    vk = torch.matmul(
        permute_final_dims(value, (1, 2, 0)),  # [*, H, C_hidden, L]
        permute_final_dims(key, (1, 0, 2)),  # [*, H, L, C_hidden]
    )

    # [*, H, C_hidden, L]
    r_mat = get_rq(query, key)

    # [*, H, C_hidden, L]
    output = torch.matmul(vk, r_mat)

    # [*, L, H, C_hidden]
    output = permute_final_dims(output, (2, 0, 1))

    return output


@torch.jit.script
def get_rq(
    query: torch.Tensor,  # [*, L, H, C_hidden]
    key: torch.Tensor,  # [*, L, H, C_hidden]
    lmbd: float = 1,
):
    batch_dims, L, h, c = query.shape
    query = permute_final_dims(query, (1, 2, 0))  # [*, H, C_hidden, L]
    key = permute_final_dims(key, (1, 2, 0))  # [*, H, C_hidden, L]

    rq = []
    # [*, H, C_hidden, K]
    r_t = lmbd * torch.eye(c)  # [C_hidden, C_hidden]

    for t in range(L):
        k_t = key[t]  # [*, H, C_hidden, 1]

        # [*, H, C_hidden, 1]
        rk = torch.matmul(r_t, k_t)  # [C_hidden, C_hidden]  # [*, H, C_hidden, 1]

        # [*, H, C_hidden, C_hidden]
        _numerator = torch.matmul(rk, k_t.transpose(-1, -2))  # [*, H, 1, C_hidden]

        # [*, H, C_hidden, C_hidden]
        _numerator = torch.matmul(
            _numerator, r_t  # [*, H, C_hidden, C_hidden]
        )  # [C_hidden, C_hidden]

        # [*, H, 1, 1]
        _denominator = 1 + torch.matmul(
            k_t.permute(-1, -2), rk  # [*, H, 1, C_hidden]  # [*, H, C_hidden, 1]
        )
        r_t = r_t - _numerator * (1 / _denominator)
        rq.append(
            # [*, H, C_hidden, 1]
            torch.matmul(
                r_t, query[..., t]  # [C_hidden, C_hidden]  # [*, H, C_hidden, 1]
            )
        )
    # L * [*, H, C_hidden, 1] -> [*, H, C_hidden, L]
    return torch.cat(rq, dim=-1)


@torch.jit.ignore
def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s
