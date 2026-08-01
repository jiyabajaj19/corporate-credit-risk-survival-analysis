# Bias-Corrected Corporate Credit Risk Survival Analysis

## Overview

This project develops a survival-analysis framework for estimating corporate default risk under delayed entry and right censoring.

Traditional credit-risk models usually estimate whether a company will default. This project instead models how default risk evolves over time and estimates when default or financial distress may occur.

The main research focus is the effect of biased sampling and delayed observation on Cox proportional-hazards estimates.

## Simulation Study

A controlled Monte Carlo study was conducted to evaluate corporate
survival models under delayed entry, right censoring, and quarterly
time-varying financial characteristics.

### Data-Generating Process

The simulation generated 800 observed firms per repetition using:

- a Weibull baseline hazard;
- delayed database entry;
- administrative right censoring;
- quarterly time-varying financial covariates;
- risk-dependent exclusion of firms that defaulted before observation.

The financial predictors included:

- leverage;
- low interest coverage;
- current ratio;
- cash ratio;
- return on assets;
- debt growth;
- revenue growth;
- operating cash-flow ratio;
- firm size;
- EBITDA margin.

The experiment was repeated 200 times.

### Models Compared

1. **Baseline-at-entry Cox model**  
   Uses financial variables from the first observed quarter and measures
   follow-up from database entry.

2. **Left-truncated baseline Cox model**  
   Accounts for delayed entry but still freezes financial variables at
   their first observed values.

3. **Time-varying Cox model**  
   Uses start-stop survival intervals and updated financial information
   for every observed quarter.

### Aggregate Results

| Model | Mean Absolute Bias | Mean RMSE | Mean 95% CI Coverage |
|---|---:|---:|---:|
| Baseline-at-entry Cox | 0.2920 | 0.3047 | 0.2170 |
| Left-truncated baseline Cox | 0.2939 | 0.3066 | 0.2180 |
| Time-varying Cox | **0.0632** | **0.1157** | **0.8080** |

The time-varying Cox model reduced mean absolute bias by approximately
78% and mean RMSE by approximately 62% relative to the
baseline-at-entry model.

### Main Finding

The results indicate that updating financial characteristics over time
is substantially more important than correcting delayed entry alone
when the underlying default hazard depends on current firm conditions.

The time-varying model performed considerably better overall, although
coverage remained weak for highly correlated predictors such as
leverage.

## Business Use Cases

- Corporate loan monitoring
- Bond credit analysis
- Probability-of-default estimation
- Time-to-default forecasting
- Portfolio credit-risk management
- Financial distress detection
- Credit-risk scenario analysis

## Planned Models

- Kaplan-Meier estimator
- Nelson-Aalen estimator
- Cox proportional-hazards model
- Left-truncated Cox model
- Bias-corrected pseudo-partial likelihood estimator
- Random Survival Forest
- Gradient-boosted survival model

## Repository Structure

```text
data/          Raw, processed, and simulated datasets
docs/          Technical documentation
notebooks/     Exploratory analysis
reports/       Figures and research outputs
scripts/       Executable scripts
src/           Reusable Python modules
tests/         Automated tests