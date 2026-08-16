import torch
import torch.nn as nn

class PseudoGridGenerator(nn.Module):
    def __init__(self, kR, dR, centered=True, device=None, dtype=torch.float32):
        super().__init__()

        self.kR = kR
        self.dR = dR
        self.centered = centered

        if self.centered:
            tau = torch.arange(
                -self.kR,
                self.kR + 1,
                device=device,
                dtype=dtype
            ) * self.dR
        else:
            tau = torch.arange(
                0,
                2 * self.kR + 1,
                device=device,
                dtype=dtype
            ) * self.dR

        # shape: (L,), where L = 2*kR + 1
        self.register_buffer("off_r", tau)

    def generate_offset_grid(self, X):
        """
        Args:
            X: input scalar field tensor of shape (B, 5, H, W)

            channels:
                X[:, 0] = scalar field
                X[:, 1] = theta
                X[:, 2] = cos(phi)
                X[:, 3] = sin(phi)
                X[:, 4] = given shell radius

        Returns:
            X_out: tensor of shape (B, 4 + L, H, W)

            channels:
                X_out[:, 0]      = scalar field
                X_out[:, 1]      = theta
                X_out[:, 2]      = cos(phi)
                X_out[:, 3]      = sin(phi)
                X_out[:, 4:4+L]  = pseudo-radii
        """

        if X.ndim != 4:
            raise ValueError(f"Expected X with shape (B, 5, H, W), got {X.shape}")

        if X.shape[1] != 5:
            raise ValueError(f"Expected 5 input channels, got {X.shape[1]}")

        B, _, H, W = X.shape

        # central/given shell radius: (B, 1, H, W)
        radii = X[:, 4:5, :, :]

        # radial offsets: (1, L, 1, 1)
        off_r = self.off_r.to(device=X.device, dtype=X.dtype).view(1, -1, 1, 1)

        # pseudo-radii: (B, L, H, W)
        offset_r = radii + off_r

        # output: (B, 4 + L, H, W)
        X_out = torch.cat([X[:, :4, :, :], offset_r], dim=1)

        return X_out

    def forward(self, X):
        return self.generate_offset_grid(X)