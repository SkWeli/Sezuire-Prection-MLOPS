# EEG Seizure MLOps — Senuda (D_BSE_23_0009)

> Ontology-Driven, Verifiable MLOps for Edge-Deployable EEG Seizure 
> Detection and Pre-Ictal Prediction

## Overview
This project develops a unified MLOps framework combining:
- 🧠 OWL/SHACL semantic validation for EEG pipelines
- 🔁 Reproducible experiment tracking (MLflow + DVC)
- ⚡ Edge-ready quantized models (ONNX 8-bit)

## Datasets
| Dataset | Source | Usage |
|---|---|---|
| CHB-MIT | PhysioNet | Primary training |
| Siena EEG | PhysioNet | Cross-dataset testing |
| Bonn University | neurophysicsbonn.de | Prototyping |

## Setup
```bash
git clone https://github.com/<your-username>/eeg-seizure-mlops
cd eeg-seizure-mlops
conda create -n fyp python=3.10
conda activate fyp
pip install -r requirements.txt
dvc pull
