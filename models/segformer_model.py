"""
SegFormer Model Definition

This module defines the SegFormer-B2 architecture used for
binary landslide segmentation.

Author: Rishikesh Gopal
Project: AI-Powered Landslide Assessment using Multi-Temporal InSAR Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation


class SegFormerModel(nn.Module):
    """
    SegFormer-B2 model for binary semantic segmentation.
    """

    def __init__(
        self,
        pretrained_model: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels: int = 1,
        dropout_rate: float = 0.10,
    ):

        super().__init__()

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(

            pretrained_model,

            num_labels=num_labels,

            ignore_mismatched_sizes=True

        )

        self.dropout = nn.Dropout2d(

            p=dropout_rate

        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor

            Shape:
                (B,3,H,W)

        Returns
        -------
        torch.Tensor

            Binary segmentation logits.
        """

        outputs = self.segformer(

            pixel_values=x

        )

        logits = outputs.logits

        logits = self.dropout(

            logits

        )

        logits = F.interpolate(

            logits,

            size=x.shape[-2:],

            mode="bilinear",

            align_corners=False

        )

        return logits


if __name__ == "__main__":

    dummy = torch.randn(

        1,

        3,

        100,

        100

    )

    model = SegFormerModel()

    output = model(

        dummy

    )

    print("Input Shape :", dummy.shape)

    print("Output Shape:", output.shape)