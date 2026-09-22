
import numpy as np
from daltonlens import convert, simulate


_M_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.9504559270516716, 1.0, 1.0890577507598784]) 

_EPS = 216 / 24389
_KAPPA = 24389 / 27


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1 / 2.4) - 0.055)


def linear_to_lab(lin):
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_linear(lab):
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA)
    
    yr = np.where(lab[..., 0] > _KAPPA * _EPS, fy ** 3, lab[..., 0] / _KAPPA)
    xyz[..., 1] = yr
    return (xyz * _WHITE) @ _M_XYZ2RGB.T


def srgb_to_lab(rgb):
    return linear_to_lab(srgb_to_linear(rgb))


def lab_to_srgb(lab):
    
    return linear_to_srgb(lab_to_linear(lab))


def gamut_map(lab):
    "
    return linear_to_lab(np.clip(lab_to_linear(lab), 0.0, 1.0))


def quantise_lab(lab):
    
    rgb8 = np.round(lab_to_srgb(lab) * 255.0)
    return srgb_to_lab(rgb8 / 255.0)


def lab_to_hex(lab):
    rgb8 = np.round(lab_to_srgb(np.atleast_2d(lab)) * 255.0).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*c) for c in rgb8]


def hex_to_lab(hex_list):
    rgb = np.array([[int(h.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
                    for h in hex_list])
    return srgb_to_lab(rgb)



def ciede2000(lab1, lab2):
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.mod(np.arctan2(b1, a1p), 2 * np.pi)
    h2p = np.mod(np.arctan2(b2, a2p), 2 * np.pi)

    dLp = L2 - L1
    dCp = C2p - C1p
    zero = (C1p * C2p) == 0
    dh = h2p - h1p
    dh = np.where(dh > np.pi, dh - 2 * np.pi, dh)
    dh = np.where(dh < -np.pi, dh + 2 * np.pi, dh)
    dh = np.where(zero, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(dh / 2)

    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= np.pi, hsum / 2,
                   np.where(hsum < 2 * np.pi, (hsum + 2 * np.pi) / 2,
                            (hsum - 2 * np.pi) / 2))
    hbp = np.where(zero, hsum, hbp)

    T = (1 - 0.17 * np.cos(hbp - np.radians(30)) + 0.24 * np.cos(2 * hbp)
         + 0.32 * np.cos(3 * hbp + np.radians(6))
         - 0.20 * np.cos(4 * hbp - np.radians(63)))
    dtheta = np.radians(30) * np.exp(-((np.degrees(hbp) - 275) / 25) ** 2)
    Cbp7 = Cbp ** 7
    RC = 2 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(2 * dtheta) * RC

    tL, tC, tH = dLp / SL, dCp / SC, dHp / SH
    return np.sqrt(np.maximum(tL ** 2 + tC ** 2 + tH ** 2 + RT * tC * tH, 0.0))



def _brettel_params(deficiency):
    
    cm = convert.LMSModel_sRGB_SmithPokorny75()
    xyz = {475: [0.1421, 0.1126, 1.0419], 575: [0.8425, 0.9154, 0.0018],
           485: [0.05795, 0.1693, 0.6162], 660: [0.1649, 0.0610, 0.0000]}
    lms_neutral = cm.LMS_from_linearRGB @ np.ones(3)
    if deficiency in (simulate.Deficiency.PROTAN, simulate.Deficiency.DEUTAN):
        w1, w2 = cm.LMS_from_XYZ @ xyz[475], cm.LMS_from_XYZ @ xyz[575]
    else:
        w1, w2 = cm.LMS_from_XYZ @ xyz[485], cm.LMS_from_XYZ @ xyz[660]
    n1, n2 = np.cross(lms_neutral, w1), np.cross(lms_neutral, w2)
    n_sep = np.cross(lms_neutral, simulate.lms_confusion_axis(deficiency))
    if np.dot(n_sep, w1) < 0:
        n1, n2 = n2, n1
    H1 = simulate.plane_projection_matrix(n1, deficiency)
    H2 = simulate.plane_projection_matrix(n2, deficiency)
    M, Minv = cm.LMS_from_linearRGB, cm.linearRGB_from_LMS
    return Minv @ H1 @ M, Minv @ H2 @ M, M.T @ n_sep


_BRETTEL = {
    "protan": _brettel_params(simulate.Deficiency.PROTAN),
    "deutan": _brettel_params(simulate.Deficiency.DEUTAN),
    "tritan": _brettel_params(simulate.Deficiency.TRITAN),
}

OBSERVERS = ("normal", "protan", "deutan", "tritan")


def simulate_linear(lin, observer, severity=1.0):
    
    if observer == "normal":
        return lin
    T1, T2, n = _BRETTEL[observer]
    side = (lin @ n) >= 0
    out = np.where(side[..., None], lin @ T1.T, lin @ T2.T)
    if severity < 1.0:
        out = severity * out + (1 - severity) * lin
    return np.clip(out, 0.0, 1.0)


def simulate_lab(lab, observer, severity=1.0):
    
    if observer == "normal":
        return np.asarray(lab, dtype=np.float64)
    return linear_to_lab(simulate_linear(lab_to_linear(lab), observer, severity))
