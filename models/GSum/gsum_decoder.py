import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm


class GSumDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
    ):
        super(GSumDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = LayerNorm(decoder_layer.d_model)

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        guidance: torch.Tensor,
        guidance_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = target
        for layer in self.layers:
            x = layer(
                source=source,
                source_mask=source_mask,
                guidance=guidance,
                guidance_mask=guidance_mask,
                target=x,
                target_mask=target_mask,
            )
        x = self.norm(x)
        return x
