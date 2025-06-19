# IVIMfit

**IVIMfit** is a modular Python library designed to fit intravoxel incoherent motion (IVIM) models to diffusion-weighted MRI (DWI) signals. It includes support for monoexponential, biexponential (free and segmented), triexponential, and Bayesian model fitting. The package is designed for use in clinical and research settings, allowing users to extract physiologically meaningful diffusion parameters from DWI datasets.

---

## 📦 Installation

Install from source:

```bash
git clone https://github.com/Atakanisik/IVIMfit
cd IVIMfit
pip install -e .
```
Install from pip:

```bash
pip install ivimfit
```
**Requirements:**

- numpy  
- scipy  
- matplotlib  
- pymc  
- pytensor  


---

##  Supported Models

###  Monoexponential (ADC)

$$ S(b) = S_0 \cdot e^{-b \cdot ADC} $$

```python
import numpy as np
import matplotlib.pyplot as plt
from ivimfit.utils import plot_fit, calculate_r_squared
from ivimfit.adc import fit_adc, monoexp_model

b = np.array([0, 50, 100, 200, 400, 600, 800])
s = np.array([800, 654,543,423,328,236,121])

adc = fit_adc(b, s)
r2 = calculate_r_squared(s / s[0], monoexp_model(b, adc))

fig, ax = plot_fit(b, s, monoexp_model, [adc], model_name=f"ADC Fit (R² = {r2:.4f})")
plt.show()
```

---

###  Biexponential (Free Fit)

$$ S(b)/S_0 = f \cdot e^{-b D^*} + (1-f) \cdot e^{-b D} $$

```python
import numpy as np
import matplotlib.pyplot as plt
from ivimfit.utils import plot_fit, calculate_r_squared
from ivimfit.biexp import fit_biexp_free, biexp_model

b = np.array([0, 50, 100, 200, 400, 600, 800])
s = np.array([800, 654,543,423,328,236,121])

f, D, D_star = fit_biexp_free(b, s)
r2 = calculate_r_squared(s / s[0], biexp_model(b, f, D, D_star))

fig, ax = plot_fit(b, s, biexp_model, [f, D, D_star], model_name=f"Free Fit (R² = {r2:.4f})")
plt.show()
```

---

###  Biexponential (Segmented Fit)

- Estimates $D$ using high-b values (b>=200 default) 
- Then fits $f$, $D^*$ with fixed $D$

```python
import numpy as np
import matplotlib.pyplot as plt
from ivimfit.utils import plot_fit, calculate_r_squared
from ivimfit.segmented import fit_biexp_segmented, biexp_fixed_D_model

b = np.array([0, 50, 100, 200, 400, 600, 800])
s = np.array([800, 654,543,423,328,236,121])

f, D_fixed, D_star = fit_biexp_segmented(b, s)
r2 = calculate_r_squared(s / s[0], biexp_fixed_D_model(b, f, D_star, D_fixed))

fig, ax = plot_fit(
    b, s,
    lambda b_, f_, D_star_,D_fixed: biexp_fixed_D_model(b_, f_, D_star_, D_fixed),
    [f, D_star,D_fixed],
    model_name=f"Segmented Fit (R² = {r2:.4f})"
)
plt.show()
```

---

###  Triexponential

S(b)/S₀ = f₁ · exp(–b · D₁*) + f₂ · exp(–b · D₂*) + (1 – f₁ – f₂) · exp(–b · D)

```python
import numpy as np
import matplotlib.pyplot as plt
from ivimfit.utils import plot_fit, calculate_r_squared
from ivimfit.triexp import fit_triexp_free, triexp_model

b = np.array([0, 50, 100, 200, 400, 600, 800])
s = np.array([800, 654,543,423,328,236,121])
f1, f2, D, D1_star, D2_star = fit_triexp_free(b, s)
pred = triexp_model(b, f1, f2, D, D1_star, D2_star)
r2 = calculate_r_squared(s / s[0], pred)

fig, ax = plot_fit(b, s, triexp_model, [f1, f2, D, D1_star, D2_star], model_name=f"Tri-exponential Fit (R² = {r2:.4f})")
plt.show()
```

---

###  Bayesian (MCMC)

```python
import numpy as np
import matplotlib.pyplot as plt
from ivimfit.utils import plot_fit, calculate_r_squared
from ivimfit.bayesian import fit_bayesian
from ivimfit.biexp import biexp_model

b = np.array([0, 50, 100, 200, 400, 600, 800])
s = np.array([800, 654,543,423,328,236,121])

if __name__ == "__main__":
    f, D, D_star = fit_bayesian(b, s, draws=500, chains=2)
    r2 = calculate_r_squared(s / s[0], biexp_model(b, f, D, D_star))

    fig, ax = plot_fit(b, s, biexp_model, [f, D, D_star], model_name=f"Bayesian Fit (R² = {r2:.4f})")
    plt.show()
```

Returns posterior mean estimates for $f$, $D$, and $D^*$.

---



##  License

This project is licensed under the MIT License.

---

##  Contributing

Pull requests are welcome. Please open an issue to discuss your proposed change before submitting a PR.

---

##  Acknowledgements

This work was inspired by previous IVIM modeling efforts, including:

- [ivim](https://github.com/oscarjalnefjord/ivim) by Jalnefjord et al.  
- PyMC Bayesian modeling  
- Research on segmented and triexponential IVIM models in liver and kidney imaging

**Citation**: If you use IVIMfit, please cite it using the following DOI:
[![DOI](https://zenodo.org/badge/1000868128.svg)](https://doi.org/10.5281/zenodo.15656115)