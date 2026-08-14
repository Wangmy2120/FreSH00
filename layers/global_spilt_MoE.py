import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """Complex-valued linear layer implemented with real-valued weights.

    For W = Wr + i Wi and x = xr + i xi:
        Wx = (Wr xr - Wi xi) + i (Wr xi + Wi xr).
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, xr, xi):
        out_real = self.real(xr) - self.imag(xi)
        out_imag = self.real(xi) + self.imag(xr)
        return out_real, out_imag


class ComplexMLP(nn.Module):
    """Two-layer complex-valued MLP with complex ReLU."""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = ComplexLinear(in_features, hidden_features)
        self.fc2 = ComplexLinear(hidden_features, out_features)

    def forward(self, xr, xi):
        hr, hi = self.fc1(xr, xi)
        hr = F.relu(hr)
        hi = F.relu(hi)
        return self.fc2(hr, hi)


class FrequencyExpertSystemSplitMoEWithGlobalExpert(nn.Module):
    """HiFiMoE: hierarchical frequency-informed mixture-of-experts module.

    This implementation follows the paper equations:
      * H^(k)_m = E^(k)_m(F^(k)) and H^(k) = mean_m(H^(k)_m);
      * H^i_global = U_i(F), lambda = softmax(sigma_global(F));
      * beta = softmax(sigma_segment(F));
      * H_local = Concat(beta_k * H^(k));
      * H* = F + H_local + gamma * alpha_hat * H_global.
    """

    def __init__(self, segment_num=3, experts_per_segment=2, hidden=64,
                 global_experts_num=2, channels=None):
        super().__init__()
        self.segment_num = segment_num
        self.experts_per_segment = experts_per_segment
        self.hidden = hidden
        self.global_experts_num = global_experts_num
        self.channels = channels

        self.initialized = False
        self.experts = nn.ModuleList()
        self.global_experts = None
        self.global_gate = None
        self.segment_fusion_gate = None
        self.fusion_gate = None
        self.gamma_net = None
        self.latest_gate_weights = {}

    def lazy_init(self, C, D, device):
        if self.initialized:
            return

        self.C = C
        self.D = D
        self.segment_len = (D + self.segment_num - 1) // self.segment_num
        self.padded_len = self.segment_len * self.segment_num
        self.segment_features = C * self.segment_len
        self.full_features = C * self.padded_len

        # M dedicated experts for each of the K frequency segments.
        for _ in range(self.segment_num):
            group = nn.ModuleList()
            for _ in range(self.experts_per_segment):
                group.append(
                    ComplexMLP(self.segment_features, self.hidden, self.segment_features)
                )
            self.experts.append(group)

        # N global experts processing the complete frequency spectrum.
        self.global_experts = nn.ModuleList([
            ComplexMLP(self.full_features, self.hidden, self.full_features)
            for _ in range(self.global_experts_num)
        ])

        # Gates use the channel-averaged magnitude of the full spectrum.
        # This keeps the gate input size independent of the channel count.
        self.global_gate = nn.Sequential(
            nn.Linear(self.padded_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.global_experts_num),
        )

        self.segment_fusion_gate = nn.Sequential(
            nn.Linear(self.padded_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.segment_num),
        )

        # sigma_reweight: adaptive gating weight alpha_hat from H_global.
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.padded_len, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid(),
        )

        # gamma is a learnable scalar obtained from local and global features.
        self.gamma_net = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

        self.to(device)
        self.initialized = True

    def _magnitude(self, real, imag):
        return torch.sqrt(real * real + imag * imag + 1e-12)

    def get_gate_weights(self):
        return self.latest_gate_weights

    def forward(self, x_fft):
        # x_fft: [B, C, D] complex spectrum, D = floor(L / 2) + 1.
        xr = x_fft.real
        xi = x_fft.imag
        B, C, D = xr.shape
        self.lazy_init(C, D, x_fft.device)

        # Zero-pad the frequency axis to a length divisible by segment_num.
        pad = self.padded_len - D
        xr = F.pad(xr, (0, pad))
        xi = F.pad(xi, (0, pad))
        mag = self._magnitude(xr, xi)  # [B, C, padded_len]
        mag_pool = mag.mean(dim=1)  # [B, padded_len]

        r_segments = torch.chunk(xr, self.segment_num, dim=-1)
        i_segments = torch.chunk(xi, self.segment_num, dim=-1)

        # Multi-segment expert module: process each complex frequency segment.
        h_local_real_parts = []
        h_local_imag_parts = []
        for k in range(self.segment_num):
            rk = r_segments[k].reshape(B, -1)
            ik = i_segments[k].reshape(B, -1)

            out_r = []
            out_i = []
            for expert in self.experts[k]:
                or_, oi_ = expert(rk, ik)
                out_r.append(or_)
                out_i.append(oi_)

            hr = torch.mean(torch.stack(out_r, dim=0), dim=0)
            hi = torch.mean(torch.stack(out_i, dim=0), dim=0)
            h_local_real_parts.append(hr.reshape(B, C, self.segment_len))
            h_local_imag_parts.append(hi.reshape(B, C, self.segment_len))

        h_local_real = torch.cat(h_local_real_parts, dim=-1)
        h_local_imag = torch.cat(h_local_imag_parts, dim=-1)

        # Segment-level gate beta = softmax(sigma_segment(F)).
        beta = F.softmax(self.segment_fusion_gate(mag_pool), dim=1).unsqueeze(-1)

        # H_local = Concat(beta_k * H^(k)).
        h_local_real_chunks = torch.chunk(h_local_real, self.segment_num, dim=-1)
        h_local_imag_chunks = torch.chunk(h_local_imag, self.segment_num, dim=-1)
        h_local_real = torch.cat(
            [beta[:, k, :] * h_local_real_chunks[k] for k in range(self.segment_num)],
            dim=-1,
        )
        h_local_imag = torch.cat(
            [beta[:, k, :] * h_local_imag_chunks[k] for k in range(self.segment_num)],
            dim=-1,
        )

        # Global expert module: process the complete original spectrum F.
        xr_flat = xr.reshape(B, -1)
        xi_flat = xi.reshape(B, -1)
        global_out_r = []
        global_out_i = []
        for expert in self.global_experts:
            or_, oi_ = expert(xr_flat, xi_flat)
            global_out_r.append(or_)
            global_out_i.append(oi_)

        global_out_r = torch.stack(global_out_r, dim=1)  # [B, N, full_features]
        global_out_i = torch.stack(global_out_i, dim=1)  # [B, N, full_features]
        lam = F.softmax(self.global_gate(mag_pool), dim=1).unsqueeze(-1)  # [B, N, 1]
        h_global_real = torch.sum(global_out_r * lam, dim=1).reshape(B, C, self.padded_len)
        h_global_imag = torch.sum(global_out_i * lam, dim=1).reshape(B, C, self.padded_len)

        # Adaptive fusion weight alpha_hat = sigma_reweight(H_global).
        h_global_mag = self._magnitude(h_global_real, h_global_imag)
        h_global_pool = h_global_mag.mean(dim=1)  # [B, padded_len]
        alpha_hat = self.fusion_gate(h_global_pool)  # [B, 1]

        # gamma from local and global two-path features.
        h_local_mag = self._magnitude(h_local_real, h_local_imag)
        local_pool = h_local_mag.mean(dim=-1).mean(dim=1, keepdim=True)  # [B, 1]
        global_pool = h_global_mag.mean(dim=-1).mean(dim=1, keepdim=True)  # [B, 1]
        gamma = self.gamma_net(torch.cat([local_pool, global_pool], dim=1))  # [B, 1]

        gate = gamma * alpha_hat  # [B, 1]

        # H* = F + H_local + gamma * alpha_hat * H_global.
        h_star_real = xr + h_local_real + gate.unsqueeze(-1) * h_global_real
        h_star_imag = xi + h_local_imag + gate.unsqueeze(-1) * h_global_imag

        self.latest_gate_weights = {
            'segment': beta.detach().cpu(),
            'global': lam.detach().cpu(),
            'alpha_hat': alpha_hat.detach().cpu(),
            'gamma': gamma.detach().cpu(),
        }

        if D < self.padded_len:
            h_star_real = h_star_real[:, :, :D]
            h_star_imag = h_star_imag[:, :, :D]

        return torch.complex(h_star_real, h_star_imag)
