# Model vs Majority-Class Baseline

| Model | Accuracy | Balanced Accuracy | Threshold | Precision | Recall/Sensitivity | Specificity | F1-score | AUC | False-Positive Windows/Hour | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SeizureCNN | 0.2452 | 0.3034 | 0.2600 | 0.3241 | 0.3034 | 0.6519 | 0.1601 | 0.4648 | 1305.8669 | 349 | 797 | 3314 | 108 |
| MajorityClassBaseline | 0.9000 | 0.3333 | 0.5000 | 0.3000 | 0.3333 | 0.6667 | 0.3158 | 0.5000 | 0.0000 | 0 | 4111 | 0 | 457 |
