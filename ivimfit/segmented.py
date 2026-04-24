import numpy as np
from scipy.optimize import curve_fit


def monoexp_model(b, D):
    """
    Monoexponential diffusion model:
    S(b)/S0 = exp(-b * D)
    """
    return np.exp(-b * D)


def biexp_fixed_D_model(b, f, D_star, D_fixed):
    """
    Biexponential IVIM model with fixed D:
    S(b)/S0 = f * exp(-b * D*) + (1 - f) * exp(-b * D_fixed)
    """
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_fixed)


def prepare_signal(b_values, signal, omit_b0=False, max_b=1000):
    """
    Filter signal and b-values by max_b and omit_b0 flags.
    """
    b = np.array(b_values)
    s = np.array(signal)

    if omit_b0:
        mask = (b > 0) & (b <= max_b)
    else:
        mask = b <= max_b

    return b[mask], s[mask]


# ivimfit/segmented.py içindeki ilgili fonksiyon

def fit_biexp_segmented(b_values, signal, omit_b0=False, split_b=200, p0=None, bounds=None, bounds_D=None):
    b_all, s_all = prepare_signal(b_values, signal, omit_b0=omit_b0)

    if len(b_all) < 4:
        raise ValueError("Not enough data points for segmented IVIM fitting.")

    s_all = s_all / s_all[0]

    # Step 1: D hesabı (Yüksek b-değerleri)
    high_mask = b_all >= split_b
    b_high = b_all[high_mask]
    s_high = s_all[high_mask]

    if len(b_high) < 2:
        return [np.nan, np.nan, np.nan]

    # Varsayılan D limitleri
    if bounds_D is None:
        bounds_D = (0, 0.01)

    try:
        popt_d, _ = curve_fit(monoexp_model, b_high, s_high, bounds=bounds_D)
        D_est = popt_d[0]
    except RuntimeError:
        return [np.nan, np.nan, np.nan]

    # Step 2: f ve D* hesabı (Düşük b-değerleri)
    low_mask = b_all < split_b
    b_low = b_all[low_mask]
    s_low = s_all[low_mask]

    if len(b_low) < 3:
        return [np.nan, D_est, np.nan]

    # Varsayılan f ve D* başlangıç/limitleri
    if p0 is None:
        p0 = [0.1, 0.01]
    if bounds is None:
        bounds = ([0, 0.005], [0.3, 0.05])

    try:
        popt, _ = curve_fit(
            lambda b, f, D_star: biexp_fixed_D_model(b, f, D_star, D_est),
            b_low, s_low, p0=p0, bounds=bounds
        )
        f_est, D_star_est = popt
        return [f_est, D_est, D_star_est]
    except RuntimeError:
        return [np.nan, D_est, np.nan]
