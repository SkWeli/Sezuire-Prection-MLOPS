# Model vs Majority-Class Baseline

| Model | Accuracy | Balanced Accuracy | Threshold | Precision | Recall/Sensitivity | Specificity | F1-score | AUC | False-Positive Windows/Hour | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SeizureCNN | 0.3571 | 0.3466 | 0.3200 | 0.3374 | 0.3466 | 0.6874 | 0.2524 | 0.5224 | 946.3890 | 1225 | 1600 | 3560 | 386 |
| MajorityClassBaseline | 0.7621 | 0.3333 | 0.5000 | 0.2540 | 0.3333 | 0.6667 | 0.2883 | 0.5000 | 0.0000 | 0 | 5160 | 0 | 1611 |
