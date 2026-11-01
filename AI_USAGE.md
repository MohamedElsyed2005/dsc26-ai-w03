# AI usage

## Tools I used, and for what

I used Claude to review the assignment requirements, help implement the feature engineering and time-based split, build the scikit-learn pipeline, and suggest additional tests. I reviewed the generated code and tested the implementation locally.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I reviewed the history-feature implementation to ensure that parcels with the same timestamp do not use each other's outcomes. I also checked that the time split keeps all parcels with the same boundary timestamp in the test set. Additional tests cover timestamp ties, shuffled input order, and train/test time separation.

## How I checked that my model has no leak

I checked that only features available at parcel creation are used, excluded identifiers and post-delivery fields from the model, and verified that extra or leaky columns do not change predictions.