
---
title: 'IVIMfit: A Modular Python Library for IVIM Model Fitting and Visualization'
tags:
  - Python
  - medical imaging
  - IVIM
  - diffusion MRI
  - biexponential
  - Bayesian modeling
authors:
  - name: Atakan Işık
    orcid: 0000-0001-5433-4442
    corresponding: true
    affiliation: 1
  - name: Orhan Erdem Haberal
    orcid: 0000-0003-2788-550X
    affiliation: 1
affiliations:
 - name: Department of Biomedical Engineering, Başkent University, Türkiye
   index: 1
date: 13 June 2025
bibliography: paper.bib
---

# Summary

**IVIMfit** is an open-source Python library developed for model-based analysis of intravoxel incoherent motion (IVIM) in diffusion-weighted magnetic resonance imaging (DW-MRI). The library supports multiple IVIM fitting approaches, including monoexponential (ADC), biexponential (free and segmented), Bayesian inference using PyMC, and triexponential models. By accepting externally provided signal decay vectors as input, IVIMfit computes physiologically meaningful diffusion and perfusion parameters, which are relevant in a variety of clinical and research settings such as oncology, nephrology, and hepatology.

Unlike many existing tools that are either tightly coupled with GUI platforms or limited to specific IVIM model types, IVIMfit is designed as a modular and extensible backend library. It emphasizes clarity, reproducibility, and ease of integration into larger image analysis workflows. The package includes built-in utilities for model fitting and parameter visualization, enabling both rapid prototyping and comparative evaluation of different modeling strategies.

# Statement of Need

Intravoxel incoherent motion (IVIM) modeling provides essential insights into tissue microstructure by separating diffusion and perfusion effects in DW-MRI. It has been applied in fields such as oncology, hepatology, nephrology, and neurology [@LeBihan:1988] ,[@Paschoal:2022]. Accurate estimation of IVIM parameters (e.g., $D$, $D^*$, $f$) is difficult due to low SNR and ill-posed nonlinear fitting [@Gurney:2018], [@Barbieri:2020].

Available software options include GUI-based platforms like MITK Diffusion or Olea Sphere, which offer convenience but lack of flexibility and reproducibility. Python-based libraries like ivim [@Jalnefjord:2018], which support Bayesian MCMC fitting for biexponential models. However, these tools are limited extensibility, and integration capability.

**IVIMfit** provides a lightweight Python library that supports multiple models: monoexponential, biexponential (free and segmented), Bayesian inference using PyMC, and triexponential.While including modular fitting methods, library also includes built-in plotting for visualizing the graph of fitting and calculation for goodness of fit . This fills a key methodological gap in open-source IVIM analysis.

# Related Work

Jalnefjord et al. [@Jalnefjord:2018] introduced a Python-based toolkit supporting biexponential IVIM modeling, including Bayesian inference, but the implementation is tailored to one model and lacks modular extensibility. DIPY includes basic IVIM support but is not focused on model comparison or visualization. GUI-based platforms such as MITK Diffusion and Olea Sphere limit batch processing and are not scriptable. In contrast, IVIMfit offers extended model coverage, built-in visualization, and clean integration into research workflows.

# Methods

## 1. Monoexponential (ADC) Model

The simplest approach assumes signal attenuation follows Gaussian diffusion:

$$
S(b) = S_0 \cdot e^{-b \cdot ADC}
$$

Where $S_0$ is the signal at $b=0$, and $ADC$ is the apparent diffusion coefficient. This model ignores perfusion and is commonly used as a baseline.

## 2. Biexponential Model – Free Fitting

The full IVIM model accounts for both diffusion and pseudo-diffusion:

$$
S(b) = S_0 \cdot \left[ f \cdot e^{-b D^*} + (1 - f) \cdot e^{-b D} 
\right]
$$

Where:
- $f$ = perfusion fraction
- $D^*$ = pseudo-diffusion coefficient
- $D$ = true diffusion coefficient

Nonlinear least squares (e.g., Levenberg–Marquardt) is used to estimate all three parameters simultaneously.

## 3. Biexponential Model – Segmented Fitting

To reduce sensitivity to noise, segmented fitting splits the process:
- Step 1: Estimate $D$ from high $b$-value data ($b > 200$ s/mm²) using a log-linear fit.
- Step 2: Fix $D$ and perform nonlinear regression to estimate $f$ and $D^*$ from the full dataset.

This approach stabilizes fitting, especially in low-SNR conditions [@Lemke:2009].

## 4. Bayesian Fitting

Bayesian inference is implemented using PyMC. The posterior is defined as:

$$
P(	theta | S) \propto P(S | 	theta) \cdot P(	theta)
$$

Where $	heta = \{D, D^*, f\}$ and prior distributions are user-configurable (e.g., uniform, normal, truncated). IVIMfit supports both Markov Chain Monte Carlo (NUTS) and variational inference.

## 5. Triexponential Model

A more complex model introduces two pseudo-diffusion compartments: 

S(b)/S₀ = f₁ · exp(–b · D₁*) + f₂ · exp(–b · D₂*) + (1 – f₁ – f₂) · exp(–b · D)

This 5-parameter model offers better physiological fidelity in some tissues but is more sensitive to noise. Multi-start fitting is used to improve robustness. In the equation f₁ and f₂ represent slow and fast components of perfusion fraction while  D₁* and D₂* represents slow and fast components of psueode-diffusion

# Description and Key Features

**IVIMfit** is a modular and extensible Python library that provides a complete backend framework for fitting IVIM models to diffusion-weighted signal decay curves. It is designed for researchers who need scriptable, reproducible, and flexible tools to compare, develop, or deploy IVIM modeling strategies . Library aims at fitting methods and not include extracting b-values and signal intensities from DWI-MR images. This process has left to users. This library includes functions that not only implement fitting methods but also provide visual outputs to the user. In this way, the user can follow the reliability of the applied method without the need for additional operations.

![Comparison of IVIM fitting methods. Each curve represents the modeled decay using different algorithms over the same normalized signal decay.](Comparison%20of%20Methods.png)

This figure visually compares the behavior of monoexponential, biexponential, segmented biexponential, Bayesian, and triexponential fitting methods using synthetic decay signals.

Key features include:

- **Comprehensive model support**:
  - Monoexponential (ADC)
  - Biexponential (free and segmented)
  - Bayesian modeling with PyMC (NUTS, ADVI)
  - Triexponential model with 5-parameter fitting

- **Modular architecture**:
  - Clean, reusable functions for fitting and visualization
  - Shared input-output format
  - Easily extensible for new models or inference strategies

- **Visualization**:
  - Signal/model overlay plots
  - Residuals and parameter summaries
  - Export-ready figures

- **Flexible inference**:
  - SciPy-based nonlinear optimization
  - Bayesian posterior estimation via PyMC
  
- **Research-ready**:
  - Minimal dependencies (NumPy, SciPy, matplotlib, PyMC)
  - Jupyter and script integration
  - Well-suited for comparative modeling, validation, and publication workflows
  - Installable from pip

# Acknowledgements

We thank the developers of PyMC and SciPy for enabling robust and flexible scientific computing, and acknowledge the open-source community for building tools upon which IVIMfit is based.

# References
