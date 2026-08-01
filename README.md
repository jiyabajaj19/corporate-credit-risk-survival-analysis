# Corporate Credit Risk Survival Analysis

A research-oriented survival analysis pipeline for modeling corporate bankruptcy risk using **real SEC financial statements**, **LoPucki bankruptcy events**, and **time-varying Cox proportional hazards models**.

Instead of treating default prediction as a static binary classification problem, this project models **time until bankruptcy** using quarterly financial statements and survival analysis techniques with **time-varying covariates**, **right censoring**, and **delayed entry**.

---

## Overview

This repository provides an end-to-end framework that:

- Downloads and processes SEC Company Facts data
- Reconstructs standardized quarterly accounting statements
- Engineers credit-risk financial ratios
- Parses real bankruptcy events from the LoPucki Bankruptcy Research Database
- Builds start-stop survival datasets for bankrupt and non-bankrupt firms
- Supports simulation studies for validating survival estimators
- Prepares data for time-varying Cox proportional hazards models

The project combines both **simulation-based methodology validation** and **real-world financial data**.

---

# Project Structure

```
corporate-credit-risk-survival-analysis/

├── configs/
│   ├── bankruptcy_companies.json
│   └── companies.json
│
├── data/
│   ├── raw/
│   │   ├── sec_companyfacts/
│   │   ├── sec_companyfacts_bankrupt/
│   │   ├── sec_companyfacts_controls/
│   │   └── lopucki/
│   │
│   ├── processed/
│   └── simulated/
│
├── reports/
│   ├── figures/
│   └── simulation_study/
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── simulation/
│   └── survival/
│
├── tests/
│
├── Makefile
└── README.md
```

---

# Data Sources

## 1. SEC Company Facts

Financial statement information is obtained from the SEC Company Facts API.

Data includes:

- Total Assets
- Total Liabilities
- Debt
- Cash
- Revenue
- Operating Income
- Net Income
- Operating Cash Flow
- Interest Expense
- Depreciation & Amortization
- Stockholders' Equity

Financial statements are standardized across companies before feature engineering.

---

## 2. LoPucki Bankruptcy Research Database

Real bankruptcy filing events are obtained from the LoPucki Bankruptcy Research Database.

Extracted information includes:

- Filing date
- Chapter
- CIK
- Company name
- Industry
- Assets before bankruptcy
- Liabilities before bankruptcy

These filings provide the event times used in survival analysis.

---

# Pipeline

## 1. SEC Data Processing

```
Company Facts JSON
        │
        ▼
Extract Financial Facts
        │
        ▼
Quarterly Panel Construction
        │
        ▼
Financial Ratio Engineering
        │
        ▼
Data Cleaning
```

---

## 2. Bankruptcy Event Processing

```
LoPucki Cases
      │
      ▼
Normalize CIK
      │
      ▼
Extract Filing Dates
      │
      ▼
Bankruptcy Event Table
```

---

## 3. Survival Dataset Construction

### Bankrupt Firms

```
Quarterly Features
        │
        ▼
Match Bankruptcy Event
        │
        ▼
Pre-event Quarters
        │
        ▼
Start-Stop Survival Dataset
```

Each bankrupt company contributes quarterly observations ending at its bankruptcy filing.

---

### Control Firms

Non-bankrupt firms are treated as **right-censored observations**.

```
Quarterly Features
        │
        ▼
Minimum History Filter
        │
        ▼
Start-Stop Intervals
        │
        ▼
Censored Survival Dataset
```

---

### Final Dataset

```
Bankrupt Firms
        │
        │
        ├──────────────┐
        │              │
        ▼              ▼
 Controls        Bankruptcy
        │              │
        └──────┬───────┘
               ▼
      Real Survival Dataset
```

---

# Financial Features

The project engineers several commonly used credit-risk indicators.

### Liquidity

- Current Ratio
- Cash Ratio

### Leverage

- Leverage Ratio

### Profitability

- Return on Assets
- Operating Margin
- EBITDA Margin

### Growth

- Revenue Growth
- Debt Growth

### Cash Flow

- Operating Cash Flow Ratio

### Firm Size

- Log Total Assets

### Debt Service

- Interest Coverage
- Low Interest Coverage Indicator

---

# Survival Analysis

Each company contributes multiple quarterly observations.

Example:

| Company | Quarter | Start | Stop | Event |
|----------|----------|------:|-----:|------:|
| A | Q1 | 0.00 | 0.25 | 0 |
| A | Q2 | 0.25 | 0.50 | 0 |
| A | Q3 | 0.50 | 0.75 | 0 |
| A | Q4 | 0.75 | 1.00 | 1 |

Control firms have the final observation censored (`event = 0`).

---

# Simulation Framework

The repository also contains a complete simulation framework for validating survival estimators.

Implemented components include:

- Survival time generation
- Left truncation
- Delayed entry
- Time-varying covariates
- Monte Carlo experiments
- Bias estimation
- Coverage probability estimation

Simulation outputs are saved in:

```
reports/simulation_study/
```

---

# Current Dataset

Current processed data includes:

### Bankrupt firms

- Real bankruptcy events from LoPucki
- Time-varying quarterly financial statements
- Start-stop survival intervals

### Control firms

- SEC financial statements
- Right-censored quarterly histories
- Matching feature engineering pipeline

The combined dataset contains both bankrupt and non-bankrupt firms prepared for time-varying survival modeling.

---

# Running the Pipeline

## Parse SEC Company Facts

```bash
python -m scripts.build_raw_financial_facts
```

---

## Construct Quarterly Financial Panel

```bash
python -m scripts.build_quarterly_panel
```

---

## Engineer Financial Features

```bash
python -m scripts.build_financial_features
```

---

## Clean Features

```bash
python -m scripts.clean_financial_features
```

---

## Parse LoPucki Bankruptcy Events

```bash
python -m scripts.parse_lopucki_cases
```

---

## Build Bankrupt Survival Dataset

```bash
python -m scripts.build_survival_dataset
```

---

## Build Control Survival Dataset

```bash
python -m scripts.build_control_survival_dataset
```

---

## Combine Both Groups

```bash
python -m scripts.build_real_survival_dataset
```

---

# Testing

Run the complete test suite:

```bash
python -m pytest -v
```

---

## Makefile Commands

The repository includes a Makefile to simplify common tasks.

| Command | Description |
|---------|-------------|
| `make test` | Run the complete test suite |
| `make sec` | Process SEC Company Facts into cleaned quarterly financial features |
| `make bankrupt` | Build the bankrupt-firm survival dataset |
| `make controls` | Build the control-firm survival dataset |
| `make real` | Combine bankrupt and control firms into the final survival dataset |
| `make simulation` | Run the Monte Carlo simulation study and generate summary reports |
| `make pipeline` | Execute the complete real-data survival-data pipeline |
| `make clean` | Remove generated outputs |

# Technologies

- Python
- pandas
- NumPy
- SciPy
- lifelines
- requests
- pytest
- matplotlib

---

# Repository Highlights

- Modular pipeline architecture
- Real SEC financial statement processing
- Real bankruptcy events
- Time-varying survival datasets
- Automated feature engineering
- Simulation framework for estimator validation
- Comprehensive unit testing
- Reproducible data pipeline

---

# Future Work

Planned extensions include:

- Time-varying Cox model estimation on the real dataset
- Hazard ratio analysis
- Concordance index evaluation
- Kaplan-Meier survival curves
- Coefficient visualization
- Industry-level stratified survival models
- Macroeconomic covariates
- Regularized survival models
- Benchmark comparison with binary default classifiers

---

# License

This project is intended for academic research and educational purposes.