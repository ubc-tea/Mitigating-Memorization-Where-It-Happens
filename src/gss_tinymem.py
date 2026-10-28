"""Gated Subspace Steering (GSS) for TinyMem GPT-2 models.

Method (paper Sec. 4):
  1. omega_t = log p_target(x_t | x_<t) - log p_ref(x_t | x_<t) on calibration rows.
     High-signal positions: omega_t > tau; low-signal positions: omega_t <= 0.
  2. Masked CE gradients g^hi, g^lo w.r.t. the activation h at `point` (Eq. 4).
  3. Directions:
       whitened    : M_hi = sum h g_hi^T, C = Sigma_lo + gamma I, SVD of L^-1 M_hi,
                     probe u = L^-T U~, steer v = V~                       (Eq. 7-8)
       contrastive : M_hi - M_lo = P S Q^T, probe u = Q, steer v = P,
                     signs canonicalised so the probe responds more on hi positions (Eq. 9)
  4. Gate threshold eps = q-quantile of the probe response on clean rows
     (|s| for whitened, s for contrastive).
  5. Update (Eq. 10): h' = h - sign * beta * sum_k a_k v_k,
       whitened    a_k = 1(|s_k| > eps) s_k
       contrastive a_k = [s_k - eps]_+ / max(u_k^T v_k, 1e-3)
     optionally multiplied by 1(c_t >= r), c_t = consecutive exceedances of the first probe.

The hook stays active for every forward pass, including all evaluation metrics.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class GSSConfig:
    point: str = "transformer.h.3.mlp"   # module whose output is probed and steered
    kind: str = "whitened"               # "whitened" | "contrastive"
    tau: float = 0.0                     # omega threshold for high-signal positions
    rank: int = 1
    q: float = 0.999                     # gate quantile on clean rows
    beta: float = 1.0                    # steering strength
    sign: float = 1.0                    # +1: h - beta*a*v (paper), -1: h + beta*a*v
    r: int = 1                           # consecutive-exceedance requirement
    gamma_rel: float = 1e-3              # whitening ridge, relative to tr(Sigma)/d

    def to_dict(self):
        return asdict(self)


def _module(model, point):
    return dict(model.named_modules())[point]


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@torch.no_grad()
def _token_logp(model, x):
    lp = model(x).logits[:, :-1].float().log_softmax(-1)
    return lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)


def collect_stats(model, ref, rows, point, tau=0.0, batch_size=16):
    """Accumulate M_hi, M_lo, Sigma_lo (float64) over calibration rows."""
    store = {}

    def fwd(_, __, out):
        h = _hidden(out)
        leaf = h.detach().float().requires_grad_(True)
        store["h"] = leaf
        return _replace(out, leaf.to(h.dtype))

    handle = _module(model, point).register_forward_hook(fwd)
    st = dict(M_hi=None, M_lo=None, S_lo=None, n_hi=0, n_lo=0, mean_hi=None, mean_lo=None)
    try:
        for i in range(0, rows.shape[0], batch_size):
            x = rows[i:i + batch_size]
            omega = _token_logp(model, x) - _token_logp(ref, x)
            hi, lo = omega > tau, omega <= 0
            with torch.enable_grad():
                logits = model(x).logits[:, :-1].float()
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                                     reduction="none").view_as(omega)
                h = store["h"]
                g_hi, = torch.autograd.grad((ce * hi).sum() / hi.sum().clamp_min(1), h, retain_graph=True)
                g_lo, = torch.autograd.grad((ce * lo).sum() / lo.sum().clamp_min(1), h)
            H = h.detach()[:, :-1].double()
            if st["M_hi"] is None:
                D = H.shape[-1]
                for k in ("M_hi", "M_lo", "S_lo"):
                    st[k] = torch.zeros(D, D, dtype=torch.float64, device=H.device)
                st["mean_hi"] = torch.zeros(D, dtype=torch.float64, device=H.device)
                st["mean_lo"] = torch.zeros(D, dtype=torch.float64, device=H.device)
            Hh, Hl = H[hi], H[lo]
            st["M_hi"] += Hh.T @ g_hi[:, :-1].double()[hi]
            st["M_lo"] += Hl.T @ g_lo[:, :-1].double()[lo]
            st["S_lo"] += Hl.T @ Hl
            st["mean_hi"] += Hh.sum(0)
            st["mean_lo"] += Hl.sum(0)
            st["n_hi"] += int(hi.sum())
            st["n_lo"] += int(lo.sum())
    finally:
        handle.remove()
    st["mean_hi"] /= max(1, st["n_hi"])
    st["mean_lo"] /= max(1, st["n_lo"])
    return st


def directions(st, kind, rank=1, gamma_rel=1e-3):
    """Probe u [D,K] and steer v [D,K] (float32)."""
    if kind == "whitened":
        S = st["S_lo"] / max(1, st["n_lo"])
        D = S.shape[0]
        C = S + gamma_rel * torch.trace(S) / D * torch.eye(D, dtype=S.dtype, device=S.device)
        L = torch.linalg.cholesky(C)
        Ut, _, Vh = torch.linalg.svd(torch.linalg.solve_triangular(L, st["M_hi"], upper=False))
        u = torch.linalg.solve_triangular(L.T, Ut[:, :rank], upper=True)
        v = Vh[:rank].T
    elif kind == "contrastive":
        P, _, Qh = torch.linalg.svd(st["M_hi"] - st["M_lo"])
        u, v = Qh[:rank].T.clone(), P[:, :rank].clone()
        for k in range(rank):  # probe must respond more on high-signal positions
            if st["mean_hi"] @ u[:, k] < st["mean_lo"] @ u[:, k]:
                u[:, k], v[:, k] = -u[:, k], -v[:, k]
    else:
        raise ValueError(kind)
    return u.float(), v.float()


@torch.no_grad()
def probe_response(model, point, u, kind, rows, batch_size=64):
    vals = []
    handle = _module(model, point).register_forward_hook(
        lambda m, i, o: vals.append((_hidden(o).float() @ u[:, :1]).flatten().cpu()))
    try:
        for i in range(0, rows.shape[0], batch_size):
            model.transformer(rows[i:i + batch_size])
    finally:
        handle.remove()
    v = torch.cat(vals)
    return v.abs() if kind == "whitened" else v


def quantile(x, q, max_n=2_000_000, seed=0):
    if x.numel() > max_n:
        x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(seed))[:max_n]]
    return float(torch.quantile(x.float(), q))


class GSSHook:
    """Forward hook implementing Eq. 10 with an optional consecutive gate."""

    def __init__(self, model, point, u, v, kind, eps, beta, r=1, sign=1.0):
        self.u, self.v, self.kind, self.eps = u, v, kind, eps
        self.beta, self.r, self.sign = beta, r, sign
        self.uv = (u * v).sum(0).clamp(min=1e-3)
        self.n = self.fired = 0
        self.handle = _module(model, point).register_forward_hook(self)

    def gate(self, h):
        s = h.float() @ self.u                                   # [B,T,K]
        if self.kind == "whitened":
            exceed = s.abs() > self.eps
            a = s * exceed
        else:
            exceed = s > self.eps
            a = (s - self.eps).clamp(min=0) / self.uv
        if self.r > 1:
            e = exceed[..., 0].float()
            run = torch.zeros_like(e)
            for t in range(e.shape[1]):
                run[:, t] = (run[:, t - 1] + 1) * e[:, t] if t else e[:, t]
            a = a * (run >= self.r).unsqueeze(-1)
        return a

    def __call__(self, _, __, out):
        h = _hidden(out)
        a = self.gate(h)
        fired = (a != 0).any(-1)
        self.n += fired.numel()
        self.fired += int(fired.sum())
        h2 = (h.float() - self.sign * self.beta * (a @ self.v.T)).to(h.dtype)
        return _replace(out, h2)

    def reset_counts(self):
        self.n = self.fired = 0

    @property
    def rate(self):
        return self.fired / max(1, self.n)

    def remove(self):
        self.handle.remove()


def fit(model, ref, calib_rows, clean_rows, cfg: GSSConfig):
    """Build GSS directions and the gate threshold. Returns a dict artifact."""
    st = collect_stats(model, ref, calib_rows, cfg.point, tau=cfg.tau)
    u, v = directions(st, cfg.kind, cfg.rank, cfg.gamma_rel)
    eps = quantile(probe_response(model, cfg.point, u, cfg.kind, clean_rows), cfg.q)
    return dict(config=cfg.to_dict(), u=u.cpu(), v=v.cpu(), eps=eps, n_hi=st["n_hi"], n_lo=st["n_lo"])


def attach(model, artifact, device=None):
    """Attach a fitted GSS artifact to `model`; returns the hook (call .remove() to detach).
    The directions are moved to `device`, by default the device of the model."""
    device = device or next(model.parameters()).device
    c = artifact["config"]
    return GSSHook(model, c["point"], artifact["u"].to(device), artifact["v"].to(device), c["kind"],
                   artifact["eps"], c["beta"], c["r"], c["sign"])
