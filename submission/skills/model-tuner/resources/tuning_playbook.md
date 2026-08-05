# Tuning Playbook

## Reading the initial grid

The first run spreads wide on purpose — shallow/conservative through
deep/aggressive, plus a plain baseline. If the baseline beats every tuned
config, the dataset's difficulty was probably misjudged (e.g. it's smaller
or noisier than assumed); don't force a tuned config to win, investigate the
data instead.

## Refinement rounds

After the first grid, look at the top 2-3 configs by `cv_auc_mean`:
- If they cluster in one depth/regularization region, that region is
  probably genuinely good — but keep at least one exploratory config from a
  different region in your next round anyway. On small validation sets the
  landscape can be deceptive; a close second in a different region sometimes
  overtakes the leader once seeds or folds shift slightly.
- Watch `cv_auc_std` across seeds, not just the mean. A config with a much
  higher std than its neighbors is a noisier bet, even at a similar mean.

## Ensembling: correlation, not just score

Don't just average your top-N configs by score. Load `grid_oof_predictions.csv`
and check the correlation between candidate columns:
```python
import pandas as pd
oof = pd.read_csv("grid_oof_predictions.csv")
print(oof.corr())
```
A config that scores slightly lower but is meaningfully less correlated with
your top pick is usually worth more in an ensemble than a slightly-better
config that's nearly identical to what you already have. If your top
candidates are all correlated above ~0.95, they're not adding much to each
other — pull in a config from a different band or model family before
averaging. A simple mean of 3-5 diverse configs typically beats one "best"
model, and usually beats complex stacking too, especially on small datasets
where a stacking meta-learner has too few rows to learn from reliably.

## Anti-patterns

- Don't chase `num_iterations` and `learning_rate` independently — they
  trade off against each other. This script already fixes iterations high
  and relies on early stopping, so only `learning_rate` needs comparing
  across configs.
- Don't judge a config off a single seed on a small dataset — the seed-to-seed
  variance can be larger than the actual gap between two configs. Averaging
  seeds (the `--seeds` argument) is there for this reason; use it.
- Missing values: boosting models handle `NaN` natively, often better than
  blanket imputation. Try the grid on both the raw and the
  `feature-engineer`-processed data if time allows, and compare.
