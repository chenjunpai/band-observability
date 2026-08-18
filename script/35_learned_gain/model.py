"""Learned observer gain for exp14, v2.

Architecture (FNOb-style):

    forcing = mu * classical(residual) * (1 + eps * tanh(net(sensors, residual)))

Multiplicative, not additive.  P0-2 is fixed: when the residual is zero the
classical part is zero, so the total forcing is zero regardless of what the
network's bias does.  The learned part is a bounded RELATIVE modulation of
the classical gain, which is what the contraction estimate can actually
absorb -- the v1 additive version violated its own theorem line.

The spectral shaping uses separate parameters for +kx and -kx, so the
learned kernel is not forced to be an even function of kx (v1 restriction).

The objective is `rate_loss`, the exported v2 version: log-mean over the
window by default, and the exponential weight uses GLOBAL time (t0 passed
by the trainer), so it is a genuine rate objective instead of a final-error
shortcut (P1-4).
"""

import numpy as np
import torch
import torch.nn as nn


class SensorSetEncoder(nn.Module):
    """DeepSets over {(x_i, y_i, residual_i)} -> per-sensor channel weights."""

    def __init__(self, n_freq=8, width=128, n_out=4):
        super().__init__()
        self.n_freq = n_freq
        d_in = 4 * n_freq + 1
        self.phi = nn.Sequential(nn.Linear(d_in, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                 nn.Linear(width, n_out))

    def features(self, xy):
        k = torch.arange(1, self.n_freq + 1, device=xy.device, dtype=xy.dtype)
        a, b = xy[..., :1] * k, xy[..., 1:2] * k
        return torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], dim=-1)

    def forward(self, xy, resid):
        h = self.phi(torch.cat([self.features(xy), resid], dim=-1))
        pooled = h.mean(dim=1, keepdim=True).expand_as(h)
        return self.rho(torch.cat([h, pooled], dim=-1))


class NeuralObserverGain(nn.Module):
    def __init__(self, N=128, n_ch=4, n_modes=24, eps=0.5, mu_init=20.0,
                 width=0.15, dtype=torch.float64,
                 mu_min=1.0, mu_max=100.0, width_min=0.02, width_max=1.0):
        super().__init__()
        self.N, self.n_ch, self.n_modes, self.eps = N, n_ch, n_modes, eps
        self.gain_mode = "multiplicative"
        if not (0.0 < mu_min < mu_max):
            raise ValueError("require 0 < mu_min < mu_max")
        if not (0.0 < width_min < width_max):
            raise ValueError("require 0 < width_min < width_max")
        self.mu_min, self.mu_max = float(mu_min), float(mu_max)
        self.width_min, self.width_max = float(width_min), float(width_max)
        self.enc = SensorSetEncoder(n_out=n_ch).to(dtype)
        self.log_width = nn.Parameter(
            torch.full((n_ch,), float(np.log(width)), dtype=dtype))
        cd = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.spec_pos = nn.Parameter(
            torch.randn(n_ch, n_modes, n_modes, dtype=cd) * 0.02)
        self.spec_neg = nn.Parameter(
            torch.randn(n_ch, n_modes, n_modes, dtype=cd) * 0.02)
        self.log_mu = nn.Parameter(
            torch.tensor(float(np.log(mu_init)), dtype=dtype))
        self.dtype, self.cdtype = dtype, cd
        g = torch.arange(N, dtype=dtype) * 2 * np.pi / N
        self.register_buffer("gx", g)

    def mu(self):
        lo, hi = np.log(self.mu_min), np.log(self.mu_max)
        return torch.exp(torch.clamp(self.log_mu, min=lo, max=hi))

    def widths(self):
        lo, hi = np.log(self.width_min), np.log(self.width_max)
        return torch.exp(torch.clamp(self.log_width, min=lo, max=hi))

    # -- classical part -------------------------------------------------
    @staticmethod
    def shepard(flow, mask, kernel_h, denom, resid_field):
        num = torch.fft.ifft2(torch.fft.fft2(
            (resid_field * mask).to(flow.cdtype)) * kernel_h).real
        return num / denom

    # -- learned part ---------------------------------------------------
    def scatter(self, xy, vals):
        """(B,Ns,C) sensor values -> (B,C,N,N) field, periodic Gaussian."""
        dx = self.gx[None, None, :] - xy[..., :1]
        dy = self.gx[None, None, :] - xy[..., 1:2]
        dx = torch.remainder(dx + np.pi, 2 * np.pi) - np.pi
        dy = torch.remainder(dy + np.pi, 2 * np.pi) - np.pi
        w = self.widths()[None, None, :, None]
        gx = torch.exp(-0.5 * (dx[:, :, None] / w) ** 2)
        gy = torch.exp(-0.5 * (dy[:, :, None] / w) ** 2)
        num = torch.einsum("bsc,bscx,bscy->bcxy", vals, gx, gy)
        den = torch.einsum("bscx,bscy->bcxy", gx, gy) + 1e-6
        return num / den

    def learned_field(self, xy, resid):
        f = self.enc(xy, resid)                       # (B,Ns,C)
        g = self.scatter(xy, f)                       # (B,C,N,N)
        gh = torch.fft.fft2(g)
        m = min(self.n_modes, self.N // 2)
        out = torch.zeros_like(gh[:, :1])
        # fftfreq ordering: 0, +1..+N/2-1, -N/2..-1  (kx = 0 row left zero)
        out[..., 1:m + 1, :m] = (gh[..., 1:m + 1, :m]
                                 * self.spec_pos[None, :, :m, :m]).sum(
                                     1, keepdim=True)
        out[..., -m:, :m] = (gh[..., -m:, :m]
                             * self.spec_neg[None, :, :m, :m]).sum(
                                 1, keepdim=True)
        return torch.fft.ifft2(out, s=(self.N, self.N)).real[:, 0]

    def forward(self, flow, xy, sensor_idx, mask, kernel_h, denom,
                resid_field, use_learned=True):
        """resid_field: (B,N,N) real, what - w in physical space.
        Returns the nudging forcing in SPECTRAL space, ready to be added
        to the solver right-hand side as `-forcing`."""
        B = resid_field.shape[0]
        resid_at = resid_field.reshape(B, -1)[:, sensor_idx]
        classical = self.shepard(flow, mask, kernel_h, denom, resid_field)
        if use_learned:
            learned = self.eps * torch.tanh(
                self.learned_field(xy, resid_at[..., None]))
            total = self.mu() * (classical * (1.0 + learned))
        else:
            total = self.mu() * classical
        return torch.fft.fft2(total.to(flow.cdtype)) \
            * flow.dealias.to(flow.cdtype)

    def spec(self):
        return dict(gain_mode=self.gain_mode, eps=self.eps, n_ch=self.n_ch,
                    n_modes=self.n_modes, N=self.N,
                    mu=float(self.mu().detach().cpu()),
                    mu_bounds=[self.mu_min, self.mu_max],
                    width_bounds=[self.width_min, self.width_max])


def rate_loss(err_traj, dt, t0=0.0, gamma=1.0, mode="log"):
    """err_traj: (B,T) relative errors -> scalar to minimise.

    mode="log": mean of log(err) over the window -- scale free, no gradient
        collapse at small error.  DEFAULT.
    mode="exp": exp(gamma * (t0 + k*dt)) * err^2 with GLOBAL time.  The v1
        code restarted t at 0 inside every BPTT window, so the weight never
        grew and the objective was effectively final error.
    """
    T = err_traj.shape[1]
    t = t0 + torch.arange(T, device=err_traj.device,
                          dtype=err_traj.dtype) * dt
    if mode == "log":
        return torch.log(err_traj.clamp_min(1e-30)).mean()
    if mode == "exp":
        return (torch.exp(gamma * t)[None] * err_traj ** 2).mean()
    raise ValueError(f"unknown loss mode {mode!r}")
