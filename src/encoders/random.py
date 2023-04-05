import numpy as np
import torch
from src.config.yamlize import yamlize
from src.encoders.base import BaseEncoder


@yamlize
class RandomEncoder(BaseEncoder, torch.nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""

    def __init__(
            self,
            image_channels: int = 3,
            image_height: int = 42,
            image_width: int = 144,
            z_dim: int = 32,
    ):
        super().__init__()

        self.im_c = image_channels
        self.im_h = image_height
        self.im_w = image_width
        self.z_dim = z_dim

    def encode(self, x: np.ndarray) -> torch.Tensor:
        # assume x is RGB image with shape (H, W, 3)
        v = torch.randn(1, self.z_dim)
        return v

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Generate random noise tensor of shape (batch_size, C, H, W)
        noise = torch.randn(z.shape[0], self.im_c, self.im_h, self.im_w)

        # Resize the noise tensor to the desired image size
        output = torch.nn.functional.interpolate(noise, size=(self.im_h, self.im_w))

        # Output (1, im_c, im_h, im_w)
        return output

    def update(self, data):
        # Implementation to update encoder weights based on observed data
        pass
