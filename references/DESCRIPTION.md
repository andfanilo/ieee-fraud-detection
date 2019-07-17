# Data Description

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`.

The data is broken into two files `identity` and `transaction`, which are joined by `TransactionID`. Not all transactions have corresponding identity information.

## Categorical Features - Transaction

- ProductCD
- emaildomain
- card1 - card6
- addr1, addr2
- P_emaildomain
- R_emaildomain
- M1 - M9

## Categorical Features - Identity

- DeviceType
- DeviceInfo
- id_12 - id_38

The `TransactionDT` feature is a timedelta from a given reference datetime (not an actual timestamp).

## Files

- **train\_{transaction, identity}.csv** - the training set
- **test\_{transaction, identity}.csv** - the test set (you must predict the isFraud value for these observations)
- **sample_submission.csv** - a sample submission file in the correct format
