# Fraud Risk Decision Policy

## Purpose

This policy defines how fraud probability scores should be converted into business actions for transaction risk decisioning.

## Risk Bands

### Low Risk

Transactions with fraud probability below 0.30 are considered low risk.

Recommended action:

Approve the transaction and continue standard fraud monitoring.

Reasoning:

Low-risk transactions do not show enough model-based risk to require manual review or denial.

---

### Medium Risk

Transactions with fraud probability greater than or equal to 0.30 and below 0.80 are considered medium risk.

Recommended action:

Send the transaction to manual review.

Reasoning:

Medium-risk transactions show elevated fraud probability but are not high enough for automatic denial. These cases should be reviewed by a fraud analyst.

---

### High Risk

Transactions with fraud probability greater than or equal to 0.80 are considered high risk.

Recommended action:

Deny the transaction and escalate the case to fraud operations.

Reasoning:

High-risk transactions exceed the denial threshold and are likely to contain fraud patterns detected by the model.

---

## False Positives and False Negatives

False positives occur when legitimate transactions are incorrectly flagged as fraud. This can increase customer friction and manual review workload.

False negatives occur when fraudulent transactions are incorrectly approved. This can increase fraud losses.

Thresholds should be selected based on the business tradeoff between reducing fraud losses and minimizing customer friction.

---

## Analyst Review Guidance

Manual review cases should be prioritized when the model assigns a medium or high fraud probability, especially when the transaction is near the denial threshold.

Analysts should review the model score, risk band, key risk signals, and historical fraud patterns before making a final decision.

---

## Escalation Guidance

Transactions classified as high risk should be escalated to fraud operations. These cases may require additional verification, account review, or transaction blocking.