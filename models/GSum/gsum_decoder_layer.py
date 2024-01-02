import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class GSumDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        batch_first: bool = False,
    ):
        super(GSumDecoderLayer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.first_multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.second_multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.activation = F.relu

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # first multihead attention block
    def _first_mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.first_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    # first multihead attention block
    def _second_mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.second_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout3(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        guidance: torch.Tensor,
        guidance_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(target + self._sa_block(target, target_mask))
        x = self.norm2(x + self._first_mha_block(target, source, source_mask))
        x = self.norm3(x + self._second_mha_block(target, source, source_mask))
        x = self.norm4(x + self._ff_block(x))
        return x
