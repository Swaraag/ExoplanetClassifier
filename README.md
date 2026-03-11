# Transit or Mirage: Machine Learning Classification of Kepler Planet Candidates

Launched in 2009, the Kepler Space Telescope was NASA's first mission dedicated to finding planets outside our solar system. Monitoring over 150,000 stars in the constellation Cygnus, Kepler confirmed more than 2,600 exoplanets before its retirement in October 2018.[^1]

Kepler detects planets through the transit method: when a planet passes directly between its host star and the telescope, it blocks a fraction of the star's light, producing a measurable dip in the star's light curve, calculated by

$$\delta = \left(\frac{R_p}{R_*}\right)^2$$

where $R_p$ is the planetary radius and $R_*$ is the stellar radius.[^2]

Despite Kepler's success, not every transit-like signal corresponds to a real planet. Astrophysical false positives (signals that mimic planetary transits but originate from other phenomena) frequently contaminate the candidate catalog. The most common sources are eclipsing binaries and their variants (grazing and background), which produce periodic brightness dips or diluted eclipse signals that mimic planetary transits. Across the full KOI catalog, the global false positive rate has been estimated at approximately 9.4%, though the rate varies significantly with planet size and orbital period.[^3] Furthermore, misclassification carries real costs in both directions: a missed planet may remove a candidate from consideration, while a false positive wastes expensive radial velocity follow-up time that could otherwise be spent confirming real discoveries.

Missing a real planet (a false negative) is a worse outcome, however, and the asymmetry in error costs motivates prioritizing false negative reduction over false positive reduction. This project trains and compares two ensemble classifiers, Random Forest and XGBoost, on the NASA Kepler Objects of Interest cumulative table to distinguish confirmed planets from false positives. Model predictions are interpreted using SHAP values to connect learned feature importance back to the underlying astrophysics, and both models are benchmarked against NASA's own disposition scores. The trained classifiers are then applied to 1,979 unresolved candidates to produce a prioritized target list for radial velocity follow-up, connecting directly to active research efforts extracting exoplanet signals from RV time series.

[^1]: NASA, [Kepler Mission Overview](https://science.nasa.gov/mission/kepler/)
[^2]: University of Nebraska-Lincoln, [The Transit Method](https://astro.unl.edu/newRTs/Transits/background/Transit1.html)
[^3]: Fressin et al. (2013), [False Positive Rate of Kepler](https://arxiv.org/abs/1301.0842)

## Dataset

The dataset is the [Kepler Objects of Interest (KOI) cumulative table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) from the NASA Exoplanet Archive. Each row corresponds to a transit signal detected by Kepler, labeled as `CONFIRMED` (2,746), `FALSE POSITIVE` (4,839), or `CANDIDATE` (1,979). The confirmed and false positive rows form the labeled training set; candidates are set aside for final prediction.

The CSV is included in the repository as `cumulative_2026.03.09_14.43.47.csv`.

---

## Results

| Model | Accuracy | ROC-AUC | CV AUC | False Negatives |
|---|---|---|---|---|
| Random Forest | 92.7% | 0.980 | 0.9780 ± 0.015 | 53 |
| XGBoost | 94.5% | 0.986 | 0.9807 ± 0.014 | 28 |

XGBoost outperformed Random Forest across every metric. Most notably, its false negative rate was 5.0% vs. Random Forest's 9.5%, nearly halving the rate of missed planets, which is the primary error cost asymmetry this project optimizes for.

---

## Key Findings

- XGBoost flagged **394 of 1,979 unresolved candidates** as likely planets
- **152 of those 394** fall in the sub-Neptune range ($1.7\text{--}3.5 \ R_\oplus$), peaking around ($2.0\text{--}2.1 \ R_\oplus$), consistent with known planetary occurrence rates
- **320 of 394** flagged candidates (81%) have NASA disposition scores above 0.90, indicating strong agreement with NASA's own vetting pipeline
- **34 candidates** where XGBoost and NASA disagree (scores below 0.5) are the most actionable: either the model caught something NASA's pipeline missed, or vice versa
- SHAP analysis confirmed predictions are driven by physically meaningful features: small planet radius and moderate SNR push strongly toward `CONFIRMED`, while large radius and high SNR push toward `FALSE POSITIVE`

---

## Full Writeup

A full writeup covering data, methods, results, SHAP analysis, and candidate predictions is available [here](exoplanet_classification_paper.pdf).

---

## Tech Stack

Python, scikit-learn, XGBoost, SHAP, pandas, matplotlib

## How to Run

**Dependencies**

This project requires [libomp](https://formulae.brew.sh/formula/libomp) (OpenMP runtime) for XGBoost to function correctly. On macOS:
```bash
brew install libomp
```

Then install Python dependencies:
```bash
pip install -r requirements.txt
```

Jupyter Notebook is also required to run `playground.ipynb`. Install via:
```bash
pip install jupyter
```

or use [Anaconda](https://www.anaconda.com/).

---

**Running the Models**

Train and evaluate the Random Forest classifier (prints results to terminal):
```bash
python randomforestclf.py
```

Train and evaluate the XGBoost classifier (prints results to terminal):
```bash
python xgboostclf.py
```

**Figures & Analysis**

All figures and visualizations are generated and stored in `playground.ipynb`. Launch with:
```bash
jupyter notebook playground.ipynb
```
