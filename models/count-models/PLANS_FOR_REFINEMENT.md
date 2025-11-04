## Flan-T5-base

A small number of big misses will barely move MAE but will blow up RMSE (squares amplify tail errors). Your exact-match/within-1 are very high → most predictions are spot on; RMSE≈3 means a few large outliers.

Quick checks (paste in the Flan cell after predictions):
```python
import numpy as np, pandas as pd

errs = np.abs(flant5_predictions - true_labels)
print("95th, 99th pct abs err:", np.percentile(errs, [95, 99]))
print("Top-10 abs errors:", np.sort(errs)[-10:])

# RMSE trimmed (drop top 1% largest errors)
k = int(0.01 * len(errs))
trim_idx = np.argsort(errs)[:-k] if k>0 else np.arange(len(errs))
rmse_trim = np.sqrt(np.mean((flant5_predictions[trim_idx]-true_labels[trim_idx])**2))
print("Trimmed RMSE (99%):", rmse_trim)

# Inspect worst 20 cases
worst = np.argsort(errs)[-20:][::-1]
pd.set_option("display.max_colwidth", 200)
display(pd.DataFrame({
    "true": true_labels[worst],
    "pred": np.array(flant5_predictions)[worst],
    "raw": np.array(decoded_preds)[worst],
    "summary": test_df.iloc[worst]["incident_summary"].values
}))
```

If the worst errors are high-count cases (e.g., 20–80), consider:
- Increase max_new_tokens to 12–15 so multi-digit counts aren’t truncated.
- Slightly higher num_beams (e.g., 4) to reduce occasional big misses.
- Train-time tweaks for tails: upweight rare large counts, or evaluate/report bin-wise metrics (you already compute non-zero MAE; add 6+ bin).

