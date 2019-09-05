# Data Description

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`.

The data is broken into two files `identity` and `transaction`, which are joined by `TransactionID`. Not all transactions have corresponding identity information.

## Transaction

It contains money transfer and also other gifting goods and service, like you booked a ticket for others, etc.

- TransactionDT: timedelta from a given reference datetime (not an actual timestamp). : TransactionDT first value is 86400, which corresponds to the number of seconds in a day (60 * 60 * 24 = 86400) so I think the unit is seconds. Using this, we know the data spans 6 months, as the maximum value is 15811131, which would correspond to day 183."
- TransactionAMT: transaction payment amount in USD : Some of the transaction amounts have three decimal places to the right of the decimal point. There seems to be a link to three decimal places and a blank addr1 and addr2 field. Is it possible that these are foreign transactions and that, for example, the 75.887 in row 12 is the result of multiplying a foreign currency amount by an exchange rate
- ProductCD: product code, the product for each transaction : Product isn't necessary to be a real 'product' (like one item to be added to the shopping cart). It could be any kind of service.
- card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
- addr: address : both addresses are for purchaser addr1 as billing region addr2 as billing country
- dist: distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc
- P\_ and (R\_\_) emaildomain: purchaser and recipient email domain, certain transactions don't need recipient, so Remaildomain is null
- C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked. Can you please give more examples of counts in the variables C1-15? Would these be like counts of phone numbers, email addresses, names associated with the user? I can't think of 15. Your guess is good, plus like device, ipaddr, billingaddr, etc. Also these are for both purchaser and recipient, which doubles the number.
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

id01 to id11 are numerical features for identity, which is collected by Vesta and security partners such as device rating, ip_domain rating, proxy rating, etc. Also it recorded behavioral fingerprint like account login times/failed to login times, how long an account stayed on the page, etc. All of these are not able to elaborate due to security partner T&C. I hope you could get basic meaning of these features, and by mentioning them as numerical/categorical, you won't deal with them inappropriately.

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
