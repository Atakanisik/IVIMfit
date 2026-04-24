import numpy as np
from scipy.optimize import curve_fit


def monoexp_model(b, ADC):
    """
    Monoexponential decay model for diffusion signal.

    Parameters:
        b (float or array): b-value(s)
        ADC (float): apparent diffusion coefficient

    Returns:
        normalized signal at b
    """
    return np.exp(-b * ADC)


def prepare_signal(b_values, signal, omit_b0=False, max_b=1000):
    """
    Prepare signal and b-values by applying filters.

    Parameters:
        b_values (array-like): input b-values
        signal (array-like): corresponding signal intensities
        omit_b0 (bool): if True, excludes b=0 values
        max_b (float): maximum b-value to include

    Returns:
        filtered_b (np.array), filtered_signal (np.array)
    """
    b = np.array(b_values)
    s = np.array(signal)

    # Build mask based on omit_b0 and max_b
    if omit_b0:
        mask = (b > 0) & (b <= max_b)
    else:
        mask = b <= max_b

    return b[mask], s[mask]


def fit_adc(b_values, signal, omit_b0=False,p0=None, bounds=None):
    b, s = prepare_signal(b_values, signal, omit_b0=omit_b0)

    if len(b) < 2:
        raise ValueError("Not enough b-values after filtering to fit ADC.")

    s = s / s[0]

    # Eğer dışarıdan bounds gelmediyse varsayılanı kullan
    if bounds is None:
        bounds = (0, 0.03)

    try:
        # Eğer dışarıdan p0 geldiyse p0 ile fit et
        if p0 is not None:
            popt, _ = curve_fit(monoexp_model, b, s, p0=p0, bounds=bounds)
        else:
            popt, _ = curve_fit(monoexp_model, b, s, bounds=bounds)
        return popt[0]
    except RuntimeError:
        return np.nan
