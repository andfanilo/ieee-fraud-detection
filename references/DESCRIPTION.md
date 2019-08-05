# Data Description

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`.

The data is broken into two files `identity` and `transaction`, which are joined by `TransactionID`. Not all transactions have corresponding identity information.

## Transaction

- TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
- TransactionAMT: transaction payment amount in USD
- ProductCD: product code, the product for each transaction
- card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
- addr: address
  dist: distance
- P\_ and (R\_\_) emaildomain: purchaser and recipient email domain
- C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
- D1-D15: timedelta, such as days between previous transaction, etc.
- M1-M9: match, such as names on card and address, etc.
- Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

Categorical Features:

- ProductCD
- emaildomain
- card1 - card6
- addr1, addr2
- P_emaildomain
- R_emaildomain
- M1 - M9

## Identity

Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions.

They're collected by Vesta’s fraud protection system and digital security partners.

Categorical Features:

- DeviceType
- DeviceInfo
- id_12 - id_38

## Files

- **train\_{transaction, identity}.csv** - the training set
- **test\_{transaction, identity}.csv** - the test set (you must predict the isFraud value for these observations)
- **sample_submission.csv** - a sample submission file in the correct format

## Info from discussion

[link to discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100304)

- We define reported chargeback on card, user account, associated email address and transactions directly linked to these attributes as fraud transaction (isFraud=1); If none of above is found after 120 days, then we define as legit (isFraud=0). This is applied to both train and test. The date time for provided dataset is long enough from now to believe that label is reliable.
- These payments are from different countries, including North America, Latin America, Europe. But TransactionAmt has been converted to USD.
