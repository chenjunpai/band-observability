"""Differentiable mirror of nolab.solver.KolmogorovFlow.

Byte-for-byte the same scheme as the numpy solver -- full-spectrum FFT,
2/3 dealiasing, forcing -n_f cos(n_f y), integrating-factor Heun -- so
anything trained here is evaluated by the same physics with no
discretisation excuse.  Batched over the leading dimension.

Cross-check:  python solver_torch.py
must print a relative difference at float64 round-off level.
"""

import numpy as np
import torch


class TorchFlow:
    def __init__(self, N=128, nu=5e-3, alpha=0.1, n_forcing=4, dt=0.004,
                 L=2 * np.pi, device="cpu", dtype=torch.float64):
        self.N, self.L, self.nu, self.alpha, self.dt = N, L, nu, alpha, dt
        self.device, self.dtype = device, dtype
        cd = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.cdtype = cd
        k = np.fft.fftfreq(N, d=1.0 / N) * (2 * np.pi / L)
        kx = torch.tensor(k[:, None] * np.ones((1, N)), dtype=dtype,
                          device=device)
        ky = torch.tensor(np.ones((N, 1)) * k[None, :], dtype=dtype,
                          device=device)
        self.kx, self.ky = kx, ky
        self.k2 = kx ** 2 + ky ** 2
        self.k2inv = torch.where(self.k2 > 0,
                                 1.0 / self.k2.clamp(min=1e-30),
                                 torch.zeros_like(self.k2))
        kmax = (2.0 / 3.0) * (N // 2) * (2 * np.pi / L)
        self.dealias = ((kx.abs() <= kmax) & (ky.abs() <= kmax)).to(dtype)
        x = torch.arange(N, dtype=dtype, device=device) * L / N
        self.X, self.Y = torch.meshgrid(x, x, indexing="ij")
        self.fh = torch.fft.fft2(
            -n_forcing * torch.cos(n_forcing * self.Y).to(cd))
        self.lin = -(nu * self.k2 + alpha)
        self.E = torch.exp(self.lin * dt).to(cd)
        self.ikx, self.iky = (1j * kx).to(cd), (1j * ky).to(cd)
        self.gx = x

    def velocity(self, wh):
        psih = wh * self.k2inv.to(self.cdtype)
        return (torch.fft.ifft2(self.iky * psih).real,
                torch.fft.ifft2(-self.ikx * psih).real)

    def nonlin(self, wh):
        u, v = self.velocity(wh)
        wx = torch.fft.ifft2(self.ikx * wh).real
        wy = torch.fft.ifft2(self.iky * wh).real
        return (-torch.fft.fft2((u * wx + v * wy).to(self.cdtype))
                * self.dealias.to(self.cdtype) + self.fh)

    def step(self, wh, nudge1=None, nudge2=None):
        dt, E = self.dt, self.E
        N1 = self.nonlin(wh)
        if nudge1 is not None:
            N1 = N1 + nudge1(wh)
        w1 = E * (wh + dt * N1)
        N2 = self.nonlin(w1)
        if nudge2 is not None:
            N2 = N2 + nudge2(w1)
        return E * wh + 0.5 * dt * (E * N1 + N2)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__)
                           .resolve().parents[2]))
    from nolab import KolmogorovFlow
    rng = np.random.default_rng(0)
    f = KolmogorovFlow(N=64, nu=5e-3)
    w = f.spinup(T=2.0, seed=0)
    tf = TorchFlow(N=64, nu=5e-3)
    a = f.step(w)
    b = tf.step(torch.tensor(w, dtype=torch.complex128)[None])[0].numpy()
    print("relative difference numpy vs torch, one step:",
          np.linalg.norm(a - b) / np.linalg.norm(a))
    if torch.cuda.is_available():
        tfg = TorchFlow(N=32, nu=5e-3, device="cuda", dtype=torch.float64)
        x = torch.zeros((1, 32, 32), dtype=torch.complex128, device="cuda")
        y = tfg.step(x)
        torch.cuda.synchronize()
        if not torch.isfinite(y).all():
            raise RuntimeError("CUDA smoke test produced non-finite values")
        print("CUDA smoke: OK --", torch.cuda.get_device_name(0))
