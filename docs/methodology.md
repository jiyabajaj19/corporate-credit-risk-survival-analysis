# Methodology

## Research Objective

The project evaluates whether quarterly financial information can be used to
estimate changing corporate bankruptcy risk through survival analysis.

The work contains two related components:

1. A Monte Carlo study in which true coefficients are known.
2. An empirical application using SEC Company Facts and LoPucki bankruptcy
   events.

## Simulation Study

The simulation compares:

- a naive baseline Cox model;
- a left-truncated baseline Cox model;
- a Cox model with time-varying covariates.

Performance is evaluated using coefficient bias, empirical variance, RMSE and
confidence-interval coverage.

## Empirical Data

Quarterly financial facts are extracted from SEC Company Facts JSON files.
Bankruptcy filing dates are derived from the LoPucki cases table.

Each firm is represented in start-stop format:

| firm_id | start | stop | event | covariates |
|---|---:|---:|---:|---|
| Firm A | 0.00 | 0.25 | 0 | Quarter 1 values |
| Firm A | 0.25 | 0.50 | 0 | Quarter 2 values |
| Firm A | 0.50 | 0.75 | 1 | Quarter 3 values |

Bankrupt firms receive one event on their final retained interval. Control
firms are right-censored.

## Predictor Standardization

Financial predictors are standardized before empirical model fitting. A
coefficient therefore corresponds to a one-standard-deviation change in the
associated financial ratio.

## Ridge Regularization

The ridge time-varying Cox model uses an L2 penalty:

\[
\ell(\beta)-\lambda\sum_j\beta_j^2.
\]

The penalty reduces coefficient instability caused by correlated financial
ratios and a modest number of observed bankruptcy events.

## Model Selection

Penalty selection uses stratified five-fold cross-validation at the firm
level. All rows belonging to the same firm remain in the same fold.

The validation score is calculated using the risk estimate from each firm's
final observed interval. This is a practical firm-level discrimination metric,
but it is not a complete time-dependent concordance estimator.

The selected penalty is the smallest penalty among the candidates tied for the
highest mean validation concordance.

## Interpretation

The unpenalized time-varying Cox model is retained as the primary classical
inference model. The tuned ridge model is treated as the preferred predictive
model because it achieved stronger grouped validation concordance.