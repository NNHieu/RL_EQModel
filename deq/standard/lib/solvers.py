# Modified based on the DEQ repo.

import pickle

import numpy as np
import torch

# from icecream import ic
from termcolor import colored


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -(alpha0**3) * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1**3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum("bij, bijd -> bd", x, part_Us)  # (N, threshold)
    return -x + torch.einsum("bd, bdij -> bij", xTU, part_VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum("bdij, bij -> bd", part_VTs, x)  # (N, threshold)
    return -x + torch.einsum("bijd, bd -> bij", part_Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(f, x0, threshold, eps=1e-3, stop_mode="rel", ls=False, name="unknown"):
    # bsz, total_hsize, seq_len = x0.size()
    bsz, total_hsize = x0.shape[:2]
    L = 1
    for i in x0.shape[2:]:
        L *= i
    seq_len = L

    def g(y):
        y = y.view_as(x0)
        return (f(y) - y).view(bsz, total_hsize, -1)

    dev = x0.device
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    x_est = x0.view(bsz, total_hsize, -1)  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, threshold).to(
        dev
    )  # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len).to(dev)
    update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)  # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False

    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len
    new_objective = 1e8

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += ite + 1
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)
        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = x_est.view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        new_objective = diff_dict[stop_mode]
        if new_objective < eps:
            break
        if (
            new_objective < 3 * eps
            and nstep > 30
            and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3
        ):
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, : nstep - 1], VTs[:, : nstep - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum("bij, bij -> b", vT, delta_gx)[:, None, None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, nstep - 1] = vT
        Us[:, :, :, nstep - 1] = u
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    # Fill everything up to the threshold length
    for _ in range(threshold + 1 - len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": prot_break,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }


class SolverArgs(object):
    def __init__(self, threshold, stop_mode, eps) -> None:
        self.threshold = threshold
        self.stop_mode = stop_mode
        self.eps = eps


class SolverRun(object):
    def __init__(self, f, x0, args: SolverArgs) -> None:
        self.f = f
        self.x0 = x0
        self.args = args

    def step(self):
        raise NotImplemented

    def check(self):
        raise NotImplemented

    def run(self):
        raise NotImplemented


class AndersonArgs(SolverArgs):
    def __init__(self, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode="rel", beta=1.0) -> None:
        super(AndersonArgs, self).__init__(threshold, stop_mode, eps)
        self.m = m
        self.lam = lam
        self.beta = beta


class AndersonRun(SolverRun):
    def __init__(self, f, x0, args: AndersonArgs) -> None:
        super(AndersonRun, self).__init__(f, x0, args)
        # Get the shape
        self.shape = x0.shape
        self.dtype = x0.dtype
        self.device = x0.device

        bsz, d = x0.shape[:2]
        self.bsz = bsz
        L = 1
        for i in x0.shape[2:]:
            L *= i
        self.alternative_mode = "rel" if self.args.stop_mode == "abs" else "abs"
        self.X = self.zeros_tensor(bsz, self.args.m, d * L)
        self.F = self.zeros_tensor(bsz, self.args.m, d * L)
        # x0 flat, f(x0) flat
        self.X[:, 0] = x0.reshape(bsz, -1)
        z0 = f(x0)
        self.F[:, 0] = z0.reshape(bsz, -1)
        # X[1] = f(x0), F[1] = f^2(x0)
        self.X[:, 1], self.F[:, 1] = self.F[:, 0], f(z0).reshape(bsz, -1)

        self.H = self.zeros_tensor(bsz, self.args.m + 1, self.args.m + 1)
        self.H[:, 0, 1:] = self.H[:, 1:, 0] = 1
        self.y = self.zeros_tensor(bsz, self.args.m + 1, 1)
        self.y[:, 0] = 1

        self.trace_dict = {"abs": [], "rel": []}
        self.lowest_dict = {"abs": 1e8, "rel": 1e8}
        self.lowest_step_dict = {"abs": 0, "rel": 0}

        self.k = 2

    def zeros_tensor(self, *shape):
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def step(self):
        n = min(self.k, self.args.m)

        # Cal step
        G = self.F[:, :n] - self.X[:, :n]  # G = diff
        self.H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2)) + self.args.lam * torch.eye(n, dtype=self.dtype, device=self.device)[None]
        )
        alpha = torch.solve(self.y[:, : n + 1], self.H[:, : n + 1, : n + 1])[0][:, 1 : n + 1, 0]  # (bsz x n)

        updated_idx = self.k % self.args.m
        self.X[:, updated_idx] = (
            self.args.beta * (alpha[:, None] @ self.F[:, :n])[:, 0]
            + (1 - self.args.beta) * (alpha[:, None] @ self.X[:, :n])[:, 0]
        )
        self.F[:, updated_idx] = self.f(self.X[:, updated_idx].reshape_as(self.x0)).reshape(self.bsz, -1)

    def check(self):
        updated_idx = self.k % self.args.m
        abs_diff, rel_diff = self.cal_diff(updated_idx)
        self.trace_dict["abs"].append(abs_diff)
        self.trace_dict["rel"].append(rel_diff)
        # if collect_orbit:
        #     yield X[:, k % m].view_as(x0).clone().detach(), diff_dict[mode]
        for mode in ["rel", "abs"]:
            if self.trace_dict[mode][-1] < self.lowest_dict[mode]:
                if mode == self.args.stop_mode:
                    self.lowest_xest = self.X[:, updated_idx].view_as(self.x0).clone().detach()
                self.lowest_dict[mode] = self.trace_dict[mode][-1]
                self.lowest_step_dict[mode] = self.k

    def cal_diff(self, step_idx):
        gx = (self.F[:, step_idx] - self.X[:, step_idx]).view_as(self.x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + self.F[:, step_idx].norm().item())
        return abs_diff, rel_diff

    def run(self):
        while self.k < self.args.threshold:
            self.step()
            self.check()
            self.k += 1
            if self.trace_dict[self.args.stop_mode][-1] < self.args.eps:
                for _ in range(self.args.threshold - 1 - self.k):
                    self.trace_dict[self.args.stop_mode].append(self.lowest_dict[self.args.stop_mode])
                    self.trace_dict[self.alternative_mode].append(self.lowest_dict[self.alternative_mode])
                break
        return self.lowest_xest

    def gen(self):
        self.k = 0
        yield self.X[:, 0].view_as(self.x0).clone().detach()
        self.k = 1
        yield self.X[:, 1].view_as(self.x0).clone().detach()
        self.k = 2
        while self.k < self.args.threshold:
            self.step()
            self.check()
            updated_idx = self.k % self.args.m
            yield self.X[:, updated_idx].view_as(self.x0).clone().detach()
            self.k += 1
            if self.trace_dict[self.args.stop_mode][-1] < self.args.eps:
                for _ in range(self.args.threshold - 1 - self.k):
                    self.trace_dict[self.args.stop_mode].append(self.lowest_dict[self.args.stop_mode])
                    self.trace_dict[self.alternative_mode].append(self.lowest_dict[self.alternative_mode])
                break
        return self.lowest_xest


def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode="rel", beta=1.0, collect_orbit=False, **kwargs):
    """Anderson acceleration for fixed point iteration."""
    bsz, d = x0.shape[:2]
    L = 1
    for i in x0.shape[2:]:
        L *= i

    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.solve(y[:, : n + 1], H[:, : n + 1, : n + 1])[0][:, 1 : n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        X[:, k % m].view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k
        # if collect_orbit:
        #     yield X[:, k % m].view_as(x0).clone().detach(), diff_dict[mode]
        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    out = {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    X = F = None
    return out


def anderson_gen(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode="rel", beta=1.0, **kwargs):
    """Anderson acceleration for fixed point iteration."""
    bsz, d = x0.shape[:2]
    L = 1
    for i in x0.shape[2:]:
        L *= i

    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.solve(y[:, : n + 1], H[:, : n + 1, : n + 1])[0][:, 1 : n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        X[:, k % m].view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k
        yield X[:, k % m].view_as(x0).clone().detach(), diff_dict[mode]
        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    out = {
        "result": lowest_xest,
        "latest": X[:, k % m].view_as(x0).clone().detach(),
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    X = F = None
    return out


class ForwardRun(SolverRun):
    def __init__(self, f, x0, args: SolverArgs) -> None:
        super(ForwardRun, self).__init__(f, x0, args)
        self.f = f
        self.x0 = x0
        self.alternative_mode = "rel" if self.args.stop_mode == "abs" else "abs"
        self.f0 = f(x0)
        self.trace_dict = {"abs": [], "rel": []}
        self.lowest_dict = {"abs": 1e8, "rel": 1e8}
        self.lowest_step_dict = {"abs": 0, "rel": 0}
        self.k = 1

    def step(self):
        self.x = self.f0
        self.f0 = self.f(self.x)
        gx = (self.f0 - self.x).view_as(self.x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + self.f0.norm().item())
        self.trace_dict["abs"].append(abs_diff)
        self.trace_dict["rel"].append(rel_diff)

    def check(self):
        for mode in ["rel", "abs"]:
            if self.trace_dict[mode][-1] < self.lowest_dict[mode]:
                if mode == self.args.stop_mode:
                    self.lowest_xest = self.x.view_as(self.x0).clone().detach()
                self.lowest_dict[mode] = self.trace_dict[mode][-1]
                self.lowest_step_dict[mode] = self.k

    def run(self):
        # if collect_orbit:
        #     yield x.view_as(x0).clone().detach(), diff_dict[mode]
        while self.k < self.args.threshold:
            self.step()
            self.check()
            self.k += 1
            if self.trace_dict[self.args.stop_mode][-1] < self.args.eps:
                for _ in range(self.args.threshold - 1 - self.k):
                    self.trace_dict[self.args.stop_mode].append(self.lowest_dict[self.args.stop_mode])
                    self.trace_dict[self.alternative_mode].append(self.lowest_dict[self.alternative_mode])
                break
        return self.lowest_xest

    def gen(self):
        self.k = 0
        yield self.x0
        self.k = 1
        while self.k < self.args.threshold:
            self.step()
            self.check()
            yield self.x.view_as(self.x0).clone().detach()
            self.k += 1
            if self.trace_dict[self.args.stop_mode][-1] < self.args.eps:
                for _ in range(self.args.threshold - 1 - self.k):
                    self.trace_dict[self.args.stop_mode].append(self.lowest_dict[self.args.stop_mode])
                    self.trace_dict[self.alternative_mode].append(self.lowest_dict[self.alternative_mode])
                break


def forward_iteration(f, x0, threshold=50, eps=1e-2, stop_mode="rel", collect_orbit=False, **kwargs):
    assert threshold > 0
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    f0 = f(x0)
    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(threshold):
        x = f0
        f0 = f(x)
        gx = (f0 - x).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + f0.norm().item())
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        x.view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k
        # if collect_orbit:
        #     yield x.view_as(x0).clone().detach(), diff_dict[mode]
        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    out = {
        "result": lowest_xest,
        "latest": x.view_as(x0).clone().detach(),
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    x = f0 = None
    return out


def forward_iteration_gen(f, x0, threshold=50, eps=1e-2, stop_mode="rel", **kwargs):
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    f0 = f(x0)
    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(threshold):
        x = f0
        f0 = f(x)
        gx = (f0 - x).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + f0.norm().item())
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        x.view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k
        yield x.view_as(x0).clone().detach(), diff_dict[mode]
        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    out = {
        "result": lowest_xest,
        "latest": x.view_as(x0).clone().detach(),
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    x = f0 = None
    return out


def analyze_broyden(res_info, err=None, judge=True, name="forward", training=True, save_err=True):
    """
    For debugging use only :-)
    """
    res_est = res_info["result"]
    nstep = res_info["nstep"]
    diff = res_info["diff"]
    res_info["diff_detail"]
    prot_break = res_info["prot_break"]
    trace = res_info["trace"]
    eps = res_info["eps"]
    threshold = res_info["threshold"]
    if judge:
        return nstep >= threshold or (nstep == 0 and (diff != diff or diff > eps)) or prot_break or torch.isnan(res_est).any()

    assert err is not None, "Must provide err information when not in judgment mode"
    prefix, color = ("", "red") if name == "forward" else ("back_", "blue")
    eval_prefix = "" if training else "eval_"

    # Case 1: A nan entry is produced in Broyden
    if torch.isnan(res_est).any():
        msg = colored(f"WARNING: nan found in Broyden's {name} result. Diff: {diff}", color)
        print(msg)
        if save_err:
            pickle.dump(err, open(f"{prefix}{eval_prefix}nan.pkl", "wb"))
        return (1, msg, res_info)

    # Case 2: Unknown problem with Broyden's method (probably due to nan update(s) to the weights)
    if nstep == 0 and (diff != diff or diff > eps):
        msg = colored(f"WARNING: Bad Broyden's method {name}. Why?? Diff: {diff}. STOP.", color)
        print(msg)
        if save_err:
            pickle.dump(err, open(f"{prefix}{eval_prefix}badbroyden.pkl", "wb"))
        return (2, msg, res_info)

    # Case 3: Protective break during Broyden (so that it does not diverge to infinity)
    if prot_break and np.random.uniform(0, 1) < 0.05:
        msg = colored(
            f"WARNING: Hit Protective Break in {name}. Diff: {diff}. Total Iter: {len(trace)}",
            color,
        )
        print(msg)
        if save_err:
            pickle.dump(err, open(f"{prefix}{eval_prefix}prot_break.pkl", "wb"))
        return (3, msg, res_info)

    return (-1, "", res_info)
