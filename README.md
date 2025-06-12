# IVIMfit

**IVIMfit** is a modular Python library designed to fit intravoxel incoherent motion (IVIM) models to diffusion-weighted MRI (DWI) signals. It includes support for monoexponential, biexponential (free and segmented), triexponential, and Bayesian model fitting. The package is designed for use in clinical and research settings, allowing users to extract physiologically meaningful diffusion parameters from DWI datasets.

---

## 📦 Installation

Install from source:

```bash
git clone https://github.com/yourusername/ivimfit_lib.git
cd ivimfit_lib
pip install -e .
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
from ivimfit.adc import fit_adc
result = fit_adc(bvals, signals)
```

---

###  Biexponential (Free Fit)

$$ S(b)/S_0 = f \cdot e^{-b D^*} + (1-f) \cdot e^{-b D} $$

```python
from ivimfit.biexp import fit_biexp_free
result = fit_biexp_free(bvals, signals)
```

---

###  Biexponential (Segmented Fit)

- Estimates $D$ using high-b values  
- Then fits $f$, $D^*$ with fixed $D$

```python
from ivimfit.segmented import fit_biexp_segmented
result = fit_biexp_segmented(bvals, signals)
```

---

###  Triexponential

$$ \frac{S(b)}{S_0} = f_1 e^{-b D_1^*} + f_2 e^{-b D_2^*} + (1 - f_1 - f_2) e^{-b D}$$

```python
from ivimfit.triexp import fit_triexp_free
result = fit_triexp_free(bvals, signals)
```

---

###  Bayesian (MCMC)

```python
from ivimfit.bayesian import fit_bayesian
result = fit_bayesian(bvals, signals)
```

Returns posterior mean estimates for $f$, $D$, and $D^*$.

---

##  Visualization

You can visualize any model’s fit using the built-in utility:

```python
from ivimfit.utils import plot_fit
plot_fit(bvals, signals, model_func, result["params"], model_name="Biexponential")
```

- Plots both original signal and fitted curve  
- Shows $R^2$ and parameters on graph

---

##  License

This project is licensed under the MIT License.

---

##  Contributing

Pull requests are welcome. Please open an issue to discuss your proposed change before submitting a PR.

---

##  Acknowledgements

This work was inspired by previous IVIM modeling efforts, including:

- [ivim](https://github.com/DevelopmentalImagingMCRI/ivim) by Jalnefjord et al.  
- PyMC Bayesian modeling  
- Research on segmented and triexponential IVIM models in liver and kidney imaging
