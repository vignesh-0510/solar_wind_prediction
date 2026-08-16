import torch
import torch.nn.functional as F
import math


# -----------------------------
# Physical constants in CGS
# -----------------------------
C_LIGHT = 2.99792458e10       # cm / s
R_SUN = 6.96e10               # cm
M_PROTON = 1.67262192369e-24    # g
K_BOLTZMANN = 1.380649e-16      # erg/K

RHO_SCALE_CGS = 1.6726e-16      # g/cm^3 per MAS unit
P_SCALE_CGS = 0.3875717          # dyn/cm^2 per MAS unit
T_SCALE_K = 2.807067e7           # K per MAS unit

PRESSURE_FACTOR = 2.0
TEMP_CODE_FACTOR = (
    P_SCALE_CGS * M_PROTON
    / (
        PRESSURE_FACTOR
        * RHO_SCALE_CGS
        * K_BOLTZMANN
        * T_SCALE_K
    )
)
def get_temp_code_factor():
    return TEMP_CODE_FACTOR

def d_dphi_periodic(q, phi):
    """
    Periodic central derivative with respect to phi.

    q:   (B, H, W)
    phi: (W,) in radians, assumed uniformly spaced over longitude
    """
    dphi = phi[1] - phi[0]
    return (torch.roll(q, shifts=-1, dims=-1) - torch.roll(q, shifts=1, dims=-1)) / (2.0 * dphi)


def d_dtheta_nonperiodic(q, theta):
    """
    Central derivative with respect to theta.
    Uses one-sided differences at theta boundaries.

    q:     (B, H, W)
    theta: (H,) in radians
    """
    dq = torch.zeros_like(q)

    # interior central difference
    dtheta_mid = theta[2:] - theta[:-2]  # (H-2,)
    dq[:, 1:-1, :] = (q[:, 2:, :] - q[:, :-2, :]) / dtheta_mid[None, :, None]

    # forward difference at lower boundary
    dq[:, 0, :] = (q[:, 1, :] - q[:, 0, :]) / (theta[1] - theta[0])

    # backward difference at upper boundary
    dq[:, -1, :] = (q[:, -1, :] - q[:, -2, :]) / (theta[-1] - theta[-2])

    return dq


def radial_current_from_curl(B_theta, B_phi, theta, phi, r_cm):
    """
    Compute J_r from B_theta and B_phi on a spherical shell.

    B_theta: (B, H, W), CGS Gauss
    B_phi:   (B, H, W), CGS Gauss
    theta:   (H,), radians, colatitude
    phi:     (W,), radians, longitude
    r_cm:    scalar radius in cm

    Returns:
        J_r_curl: (B, H, W), CGS statA / cm^2
    """
    theta = theta.to(B_theta.device).to(B_theta.dtype)
    phi = phi.to(B_theta.device).to(B_theta.dtype)

    sin_theta = torch.sin(theta).clamp_min(1e-6)  # avoid pole singularity

    # term 1: d/dtheta (sin(theta) * B_phi)
    sin_Bphi = sin_theta[None, :, None] * B_phi
    dtheta_sin_Bphi = d_dtheta_nonperiodic(sin_Bphi, theta)

    # term 2: dB_theta / dphi
    dphi_Btheta = d_dphi_periodic(B_theta, phi)

    curl_B_r = (
        dtheta_sin_Bphi - dphi_Btheta
    ) / (r_cm * sin_theta[None, :, None])

    J_r_curl = (C_LIGHT / (4.0 * math.pi)) * curl_B_r

    return J_r_curl


def radial_current_physics_loss(
    pred_cgs,
    theta,
    phi,
    r_solar=30.742662,
    reduction="mean",
    relative=True,
    eps=1e-18,
):
    """
    Physics residual loss enforcing:
        J_r_pred ≈ c/(4π) * (curl B)_r

    pred_cgs: (B, 9, H, W), physical CGS prediction
              channel order: [VT, VP, BT, BP, JT, JP, JR, RHO, P]

    theta:    (H,), radians
    phi:      (W,), radians
    r_solar:  shell radius in solar radii

    Returns:
        loss, diagnostics
    """
    B_theta_pred = pred_cgs[:, 2]  # BT
    B_phi_pred = pred_cgs[:, 3]    # BP
    J_r_pred = pred_cgs[:, 6]      # JR

    r_cm = r_solar * R_SUN

    J_r_curl = radial_current_from_curl(
        B_theta=B_theta_pred,
        B_phi=B_phi_pred,
        theta=theta,
        phi=phi,
        r_cm=r_cm,
    )

    residual = J_r_pred - J_r_curl

    if relative:
        # normalize residual so the physics loss is numerically stable
        denom = torch.sqrt(torch.mean(J_r_pred.detach() ** 2)) + eps
        residual = residual / denom

    if reduction == "mean":
        loss = torch.mean(residual ** 2)
    elif reduction == "sum":
        loss = torch.sum(residual ** 2)
    else:
        loss = residual ** 2

    diagnostics = {
        "jr_physics_loss": loss.detach(),
        "jr_pred_rms": torch.sqrt(torch.mean(J_r_pred.detach() ** 2)),
        "jr_curl_rms": torch.sqrt(torch.mean(J_r_curl.detach() ** 2)),
        "jr_residual_rms": torch.sqrt(torch.mean((J_r_pred.detach() - J_r_curl.detach()) ** 2)),
    }

    return loss, diagnostics

def positivity_constraint_loss(pred_cgs, reduction="mean"):
    """
    Enforce positivity of density and pressure.

    pred_cgs: (B, 9, H, W), physical CGS prediction
              channel order: [VT, VP, BT, BP, JT, JP, JR, RHO, P]

    Returns:
        loss
    """
    rho_pred = pred_cgs[:, 7]  # RHO
    p_pred = pred_cgs[:, 8]    # P

    rho_violation = F.relu(-rho_pred)  # positive if rho < 0
    p_violation = F.relu(-p_pred)      # positive if p < 0

    if reduction == "mean":
        loss = torch.mean(rho_violation ** 2) + torch.mean(p_violation ** 2)
    elif reduction == "sum":
        loss = torch.sum(rho_violation ** 2) + torch.sum(p_violation ** 2)
    else:
        loss = rho_violation ** 2 + p_violation ** 2

    return loss

M_PROTON = 1.67262192369e-24   # g
K_BOLTZMANN = 1.380649e-16     # erg / K

def implied_temperature_mas(
    pred_mas,
    rho_floor=1e-10,
):
    """
    Calculate implied temperature in MAS code units.

    pred_mas:
        (B, 9, H, W)
        [VT, VP, BT, BP, JT, JP, JR, RHO, P]

    IMPORTANT:
        pred_mas must be:
          - denormalized
          - inverse transformed
          - NOT converted to CGS
    """

    rho = pred_mas[:, 7]
    pressure = pred_mas[:, 8]

    rho_safe = torch.clamp(rho, min=rho_floor)

    temp_mas = (
        TEMP_CODE_FACTOR
        * pressure
        / rho_safe
    )

    return temp_mas

def implied_temperature_loss(pred_mas, temp_true_mas, rho_floor=1e-10, eps=1e-10, reduction="mean"):
    temp_calculated = implied_temperature_mas(pred_mas, rho_floor=rho_floor)

    relative_error = (temp_calculated - temp_true_mas) / (temp_true_mas + eps)
    if reduction == "mean":
        loss = (relative_error ** 2).mean()
    elif reduction == "sum":
        loss = (relative_error ** 2).sum()
    else:
        loss = (relative_error ** 2)
    return loss
