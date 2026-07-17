# Model vs Majority-Class Baseline

| Model | Accuracy | Balanced Accuracy | Threshold | Precision | Recall/Sensitivity | Specificity | F1-score | AUC | False-Positive Windows/Hour | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SeizureCNN | 0.5788 | 0.2943 | 0.5900 | 0.2957 | 0.2943 | 0.6402 | 0.2950 | 0.5301 | 366.3270 | 299 | 3782 | 1378 | 1312 |
| MajorityClassBaseline | 0.7621 | 0.3333 | 0.5000 | 0.2540 | 0.3333 | 0.6667 | 0.2883 | 0.5000 | 0.0000 | 0 | 5160 | 0 | 1611 |
