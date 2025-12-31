from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def gmm_nll_loss(y_true: torch.Tensor, pi: torch.Tensor,
                 mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    batch_size, n_modes, _ = mu.shape
    y_expanded = y_true.unsqueeze(1).expand_as(mu)
    log_2pi = np.log(2 * np.pi)
    diff = (y_expanded - mu) / sigma
    log_gaussian = -0.5 * (log_2pi + 2 * torch.log(sigma) + diff ** 2)
    log_gaussian = log_gaussian.sum(dim=-1)
    log_pi = torch.log(pi + 1e-10)
    log_weighted = log_pi + log_gaussian
    log_prob = torch.logsumexp(log_weighted, dim=-1)
    return -log_prob.mean()


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    boundary_dist: torch.Tensor,
    boundary_zone: torch.Tensor,
    w_nll: float = 1.0,
    w_final: float = 5.0,
    w_aux_dist: float = 0.3,
    w_aux_zone: float = 0.3,
    w_gate: float = 0.5,
    w_isolation: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    infield_mask = (boundary_zone == 0)
    boundary_mask = (boundary_zone != 0)
    goalline_mask = (boundary_zone == 3)

    pi_mixed = outputs["pi"]
    mu_mixed = outputs["mu"]
    sigma_mixed = outputs["sigma"]

    gmm_nll = gmm_nll_loss(y_true, pi_mixed, mu_mixed, sigma_mixed)
    mse_final = F.mse_loss(outputs["y_final"], y_true)

    pi_A, mu_A, sigma_A = outputs["pi_A"], outputs["mu_A"], outputs["sigma_A"]
    pi_B, mu_B, sigma_B = outputs["pi_B"], outputs["mu_B"], outputs["sigma_B"]

    if infield_mask.any():
        nll_A = gmm_nll_loss(
            y_true[infield_mask],
            pi_A[infield_mask],
            mu_A[infield_mask],
            sigma_A[infield_mask],
        )
    else:
        nll_A = torch.tensor(0.0, device=y_true.device)

    BOUNDARY_WEIGHT = 3.0
    GOALLINE_WEIGHT = 5.0
    nll_B = torch.tensor(0.0, device=y_true.device)

    sideline_mask = boundary_mask & ~goalline_mask
    if sideline_mask.any():
        nll_B_side = gmm_nll_loss(
            y_true[sideline_mask],
            pi_B[sideline_mask],
            mu_B[sideline_mask],
            sigma_B[sideline_mask],
        )
        nll_B = nll_B + BOUNDARY_WEIGHT * nll_B_side

    if goalline_mask.any():
        nll_B_gl = gmm_nll_loss(
            y_true[goalline_mask],
            pi_B[goalline_mask],
            mu_B[goalline_mask],
            sigma_B[goalline_mask],
        )
        nll_B = nll_B + GOALLINE_WEIGHT * nll_B_gl

    isolation_loss = nll_A + nll_B

    aux_dist_loss = F.smooth_l1_loss(outputs["aux_dist"], boundary_dist)
    aux_zone_loss = F.cross_entropy(outputs["aux_zone"], boundary_zone, label_smoothing=0.1)

    gate = outputs["gate"]
    gate_target = boundary_mask.float().unsqueeze(1)
    gate_loss = F.binary_cross_entropy(gate, gate_target)

    with torch.no_grad():
        max_mode = pi_mixed.argmax(dim=1)
        mode0_ratio = (max_mode == 0).float().mean().item()
        mode1_ratio = (max_mode == 1).float().mean().item()
        dist_final = torch.sqrt(((outputs["y_final"] - y_true) ** 2).sum(dim=1)).mean().item()
        dist_avg = torch.sqrt(((outputs["y_avg"] - y_true) ** 2).sum(dim=1)).mean().item()

    total_loss = (
        w_nll * gmm_nll +
        w_final * mse_final +
        w_isolation * isolation_loss +
        w_aux_dist * aux_dist_loss +
        w_aux_zone * aux_zone_loss +
        w_gate * gate_loss
    )

    loss_dict = {
        "gmm_nll": gmm_nll.item(),
        "mse_final": mse_final.item(),
        "nll_A": nll_A.item() if isinstance(nll_A, torch.Tensor) else nll_A,
        "nll_B": nll_B.item() if isinstance(nll_B, torch.Tensor) else nll_B,
        "isolation": isolation_loss.item(),
        "gate": gate_loss.item(),
        "mode0_ratio": mode0_ratio,
        "mode1_ratio": mode1_ratio,
        "dist_final": dist_final,
        "dist_avg": dist_avg,
        "total": total_loss.item(),
    }

    return total_loss, loss_dict
