# Data Leakage Checklist

Data leakage during feature engineering is one of the most common causes of
overly optimistic performance estimates during local validation, followed by
catastrophic failure on the private leaderboard. When performing feature
engineering, strictly adhere to the following principles.

## Target Leakage Prevention

- **Rule**: Ensure no feature is directly derived from, or highly correlated
  with, the target column in a way that would not be available at true
  inference time.
- **Example**: In a loan-approval task, a column like "days between
  application and approval" is a red flag — that value doesn't meaningfully
  exist for a rejected application, so its presence or absence quietly
  encodes the answer itself.
- **Check**: For every engineered feature, ask "would this value actually
  exist at the moment I need to make a prediction, before the true outcome
  is known?" If not, drop it.

## Honest Cross-Validation

- **Rule**: Any preprocessing step that learns something from data (an
  imputer's fill value, a scaler's mean/std, a target encoder's category
  means) must be fit only on the training fold, never on the validation
  fold.
- **Why it matters**: Fitting on combined train+validation data lets the
  validation fold's own statistics quietly influence the values used to
  score it, making your cross-validation score look better than the model
  actually deserves.
- **Check**: When cross-validating, refit every preprocessing step from
  scratch inside each fold — don't fit once on the full training set and
  reuse it across folds.
