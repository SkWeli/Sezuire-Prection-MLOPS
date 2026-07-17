# Model vs Majority-Class Baseline

| Model | Accuracy | Balanced Accuracy | Threshold | Precision | Recall/Sensitivity | Specificity | F1-score | AUC | False-Positive Windows/Hour | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SeizureTCN | 0.1195 | 0.3377 | 0.0800 | 0.3508 | 0.3377 | 0.6711 | 0.0763 | 0.4632 | 1580.5166 | 452 | 100 | 4011 | 5 |
| MajorityClassBaseline | 0.9000 | 0.3333 | 0.5000 | 0.3000 | 0.3333 | 0.6667 | 0.3158 | 0.5000 | 0.0000 | 0 | 4111 | 0 | 457 |
