# Weekend Patch v4 — Non-Degenerate Alarm Threshold Selection

Replace these files:

- `src/evaluation/metrics.py`
- `src/evaluation/reporting.py`
- `src/training/train.py`

Add:

- `tests/test_threshold_policy_v4.py`

`main.py` and `src/evaluation/splits.py` are included for completeness and are unchanged from v3.

## Recommended smoke-test command

```powershell
python main.py `
  --dataset tusz `
  --stage train `
  --model cnn `
  --epochs 1 `
  --max-patients 10 `
  --batch-size 32 `
  --seed 42 `
  --alarm-threshold-policy balanced_accuracy `
  --run-name-suffix alarm_balanced `
  2>&1 | Tee-Object -FilePath "reports\weekend_cnn_alarm_balanced.log"
```

The recommended default is `balanced_accuracy`. Other supported policies:

- `youden_j`
- `f1` (legacy comparison only; may select an all-alarm threshold)
- `specificity_constrained --min-alarm-specificity 0.80`
