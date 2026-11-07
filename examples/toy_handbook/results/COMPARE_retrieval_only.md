# Comparison: B = dense_k3 vs A = bm25_k3

Differences are B - A in percentage points with a paired bootstrap 95% CI, computed over questions scored in both runs. A CI that includes zero means the difference is not clear at this sample size.

## Overall

- Hit@3 +8.3 pts [95% CI +0.0, +25.0] (n=12): the CI includes zero, so the difference is not clear at this sample size.
- Recall@3 +8.3 pts [95% CI +0.0, +25.0] (n=12): the CI includes zero, so the difference is not clear at this sample size.
- MRR +8.3 pts [95% CI -8.3, +29.2] (n=12): the CI includes zero, so the difference is not clear at this sample size.
- Precision@3 +2.8 pts [95% CI +0.0, +8.3] (n=12): the CI includes zero, so the difference is not clear at this sample size.

## qtype `lookup`

- Hit@3 +0.0 pts [95% CI +0.0, +0.0] (n=5): identical on every question.
- Recall@3 +0.0 pts [95% CI +0.0, +0.0] (n=5): identical on every question.
- MRR -10.0 pts [95% CI -30.0, +0.0] (n=5): the CI includes zero, so the difference is not clear at this sample size.
- Precision@3 +0.0 pts [95% CI +0.0, +0.0] (n=5): identical on every question.

## qtype `multi_passage`

- Hit@3 +0.0 pts [95% CI +0.0, +0.0] (n=4): identical on every question.
- Recall@3 +0.0 pts [95% CI +0.0, +0.0] (n=4): identical on every question.
- MRR +0.0 pts [95% CI +0.0, +0.0] (n=4): identical on every question.
- Precision@3 +0.0 pts [95% CI +0.0, +0.0] (n=4): identical on every question.

## qtype `paraphrase`

- Hit@3 +33.3 pts [95% CI +0.0, +100.0] (n=3): the CI includes zero, so the difference is not clear at this sample size.
- Recall@3 +33.3 pts [95% CI +0.0, +100.0] (n=3): the CI includes zero, so the difference is not clear at this sample size.
- MRR +50.0 pts [95% CI +0.0, +100.0] (n=3): the CI includes zero, so the difference is not clear at this sample size.
- Precision@3 +11.1 pts [95% CI +0.0, +33.3] (n=3): the CI includes zero, so the difference is not clear at this sample size.
