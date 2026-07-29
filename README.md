# Bias-Corrected Corporate Credit Risk Survival Analysis

## Overview

This project develops a survival-analysis framework for estimating corporate default risk under delayed entry and right censoring.

Traditional credit-risk models usually estimate whether a company will default. This project instead models how default risk evolves over time and estimates when default or financial distress may occur.

The main research focus is the effect of biased sampling and delayed observation on Cox proportional-hazards estimates.

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