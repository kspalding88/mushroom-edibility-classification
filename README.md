# Mushroom Edibility Classification

Machine learning project predicting whether a mushroom is edible or poisonous
from observable physical traits, with a safety-first focus on the features a
beginner forager could actually see in the field.

## Overview

This project builds and compares classification models that predict mushroom
edibility (edible vs. poisonous) from physical characteristics. Because calling
a poisonous mushroom "edible" is a life-safety error, the project treats
**recall on the poisonous class** — not raw accuracy — as the metric that
matters most, and tunes the decision threshold toward caution. Three classifiers
are compared:

- Logistic Regression (interpretable linear baseline)
- Decision Tree (human-readable yes/no rules)
- Random Forest (non-linear ensemble approach)

## Dataset

Dataset: secondary mushroom dataset (`secondary_data.csv`), semicolon-separated.
The data contains tens of thousands of mushroom records described by physical
traits.

Target variable:

- `class` — poisonous (`p`) vs. edible (`e`)

Class balance:

- Approximately 55.5% poisonous, 44.5% edible (no severe imbalance)

Feature groups (examples):

- Cap: cap-diameter, cap-shape, cap-color
- Stem: stem-height, stem-width, stem-color, has-ring
- Other: does-bruise-or-bleed, habitat, season

## Target Variable & Feature Selection

- Target (y): edibility, encoded so that **poisonous = 1** and edible = 0, so
  recall directly measures the share of poisonous mushrooms correctly flagged.
- Predictors (X): all remaining physical-trait columns after cleaning,
  one-hot encoded for modeling.

A second, restricted feature set is also used for a "beginner forager" test
(see Methods): only traits a novice can easily judge — cap-diameter, stem-height,
stem-width, cap-shape, cap-color, does-bruise-or-bleed, stem-color, has-ring,
habitat, and season.

## Methods of Analysis

This is a supervised binary classification problem (predicting a discrete
edible/poisonous label).

### Model 1 — Logistic Regression

A linear baseline (`max_iter=1000`). Simple and interpretable, it establishes a
reference point for how well a straightforward model separates the two classes.

### Model 2 — Decision Tree

A single decision tree (`max_depth=8`) that produces readable yes/no splitting
rules, useful for understanding which traits drive the edible/poisonous
decision.

### Model 3 — Random Forest

A Random Forest (`n_estimators=200`) to capture non-linear relationships and
trait interactions. Random Forest is appropriate here because the features are
almost entirely categorical with many levels, and averaging across many trees
reduces noise and improves stability.

## Preprocessing / Data Preparation

The preparation pipeline used a multi-step approach:

1. Missingness audit: columns examined for the percentage of missing values.
2. Dropping near-empty columns: columns more than ~80% empty were removed
   (`stem-root`, `veil-type`, `veil-color`, `spore-print-color`), since
   imputing them would be mostly guessing.
3. Filling sparse blanks: remaining categorical gaps filled with `"unknown"`
   to preserve every row rather than discard data.
4. Removing duplicates: exact duplicate records dropped.
5. Encoding: categorical traits one-hot encoded with `pd.get_dummies` so the
   models can use them.

## Validation Strategy

- Train/Test Split: 75/25 with a fixed random state (42) for reproducibility.
- The same split and seed are reused across models for fair comparison.

## Evaluation Metrics

Models were evaluated with safety as the priority:

- Recall (poisonous class): the share of poisonous mushrooms correctly caught —
  the most important metric, since a missed poisonous mushroom is the dangerous
  error.
- Accuracy: overall share of correct predictions.
- Confusion matrix: exposes the count of "dangerous misses" (poisonous called
  edible).
- ROC / AUC: how well the model ranks poisonous above edible across all
  thresholds.

## Results & Findings

### Model comparison

All three models were trained and compared on accuracy and poisonous-class
recall, with the Random Forest serving as the primary model for the deeper
safety analysis. The confusion matrix for the Random Forest is used to count
dangerous misses (poisonous mushrooms predicted edible).

### Safety-first threshold tuning

Rather than accept the default 0.50 decision cutoff, the project sweeps lower
cutoffs (0.50 → 0.30 → 0.20 → 0.10 → 0.05) and reports, at each cutoff, how many
poisonous mushrooms are caught versus how many edible mushrooms are falsely
flagged. In a life-safety setting, a few false alarms are preferable to a single
missed poisonous mushroom, so a more cautious cutoff is justified.

### Beginner-features-only test

A separate Random Forest trained only on easily observed traits tests whether
reliable prediction is possible without expert-level observations — the
realistic scenario for a novice forager. This model is the one saved for reuse,
since a practical field tool can only rely on traits a normal person can see.

## Interpretation (feature importance)

The Random Forest's feature importances are ranked to show which traits most
influence the edible/poisonous prediction, and the top 10 are plotted. This
highlights the physical characteristics most associated with toxicity in the
data.

## Outputs

Running the script produces the following figures (saved to `figures/`):

- `fig1_class_balance.png` — edible vs. poisonous counts
- `fig2_cap_diameter.png` — cap diameter by edibility
- `fig3_confusion_matrix.png` — Random Forest confusion matrix
- `fig5_roc_curve.png` — ROC curve with AUC (safety tradeoff)
- `fig4_feature_importance.png` — top 10 most important features

It also saves the trained beginner-features model (`mushroom_model.pkl`) and its
column list (`model_columns.pkl`) so predictions can be made later without
retraining, and demonstrates scoring a single new mushroom with a three-way
output: edible, poisonous, or "NOT CERTAIN" when the model is not confident.

## Conclusion

A Random Forest classifier reliably distinguishes poisonous from edible
mushrooms using observable physical traits. Framing the problem around recall of
the poisonous class — and tuning the decision threshold toward caution — reflects
the real-world cost of a wrong "edible" call. The beginner-features test shows
how much predictive power remains when the model is restricted to traits a
novice forager can actually observe, which is the basis for a practical,
safety-conscious field tool.
```

