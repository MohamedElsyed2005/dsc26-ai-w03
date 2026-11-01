# Results: Tawseel failed-delivery baseline

## Baseline table

Evaluation on the last 20% of parcels by time, using `make_parcels(4000, seed=0)`.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.5000 | 0.2200 | 0.1717 | 0.2200 |
| Rule: Not evaluated | — | — | — | — |
| My model | 0.6386 | 0.3282 | 0.1650 | 0.2200 |

## Three sentences for Sherif

The model can rank COD parcels by their risk of failed delivery, achieving a PR-AUC of 0.328 compared with the 0.220 baseline. This is a promising improvement on the synthetic test data, although the model's performance should be treated cautiously. It cannot guarantee accurate predictions in real delivery operations and must be validated on real historical data before deployment.