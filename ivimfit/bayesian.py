import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az  # R-hat hesaplaması için eklendi

def ivim_model(b, f, D, D_star):
    """
    IVIM biexponential model:
    S(b)/S0 = f * exp(-b * D*) + (1 - f) * exp(-b * D)
    """
    return f * pt.exp(-b * D_star) + (1 - f) * pt.exp(-b * D)

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

# ivimfit/bayesian.py içindeki ilgili fonksiyon

def fit_bayesian(b_values, signal, omit_b0=False, draws=500, chains=4, cores=None, progressbar=False, custom_priors=None):
    
    b, s = prepare_signal(b_values, signal, omit_b0=omit_b0)

    if len(b) < 4:
        raise ValueError("Not enough data for Bayesian IVIM fitting.")

    s = s / s[0]
    b = b.astype("float32")
    s = s.astype("float32")

    # Varsayılan sınırları (priors) ayarla
    priors = {
        'f_min': 0.0, 'f_max': 0.3,
        'D_min': 0.0005, 'D_max': 0.003,
        'Ds_min': 0.005, 'Ds_max': 0.05
    }

    # Kullanıcı dışarıdan özel sınırlar gönderdiyse güncelle
    if custom_priors is not None:
        priors.update(custom_priors)

    with pm.Model():
        # Dinamik Priors kullanımı
        f = pm.Uniform("f", lower=priors['f_min'], upper=priors['f_max'])
        D = pm.Uniform("D", lower=priors['D_min'], upper=priors['D_max'])
        D_star = pm.Uniform("D_star", lower=priors['Ds_min'], upper=priors['Ds_max'])
        
        sigma = pm.HalfNormal("sigma", sigma=0.05)
        S_model = ivim_model(b[:, None], f, D, D_star) # ivim_model importunuz neyse o
        pm.Normal("obs", mu=S_model.flatten(), sigma=sigma, observed=s)
        trace = pm.sample(draws=draws, chains=chains, cores=cores, progressbar=progressbar, target_accept=0.9)

    # ... Sonrası mevcut kodunla aynı ...

    # Posterior ortalamalarının alınması
    f_mean = trace.posterior["f"].mean().item()
    D_mean = trace.posterior["D"].mean().item()
    D_star_mean = trace.posterior["D_star"].mean().item()

    # Sadece Quality modunda (chains > 1) R-hat hesaplanır
    r_hat_max = "N/A"
    if chains > 1:
        try:
            rhat_data = az.rhat(trace)
            r_hat_max = max(
                rhat_data["f"].values.item(),
                rhat_data["D"].values.item(),
                rhat_data["D_star"].values.item()
            )
            r_hat_max = round(r_hat_max, 3)
        except Exception:
            # Herhangi bir arviz hatasında kod çökmesin diye
            pass

    return f_mean, D_mean, D_star_mean, r_hat_max