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