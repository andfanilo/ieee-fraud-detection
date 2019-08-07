# Some advice I read on the competition

## Parameter optimisation

> I do have a question though. How do I find the least tree depth? With parameter optimisation?
>
> I use trial and error by hand. (Optimizations libraries work well too). For boosted trees, I test max_depths 3 thru 13. For LGBM, I prefer to leave the default max_depth= -1 and adjust num_leaves. For num_leaves, I test powers of 2, i.e. 2^3, 2^4, 2^5, …, 2^10, 2^11, 2^12 which is 8, 16, 32, …, 1024, 2048, 4092.
>
> Understanding the meaning of tree size (whether it be max_depth or num_leaves) helps you adjust the parameter efficiently. First tree size needs to be adjusted to match your encoding (category, numerical, one-hot, etc) as discussed above. Second, tree size controls how much interaction you allow between variables. If variables are mostly independent, smaller trees are better. If variables dependent on each other, larger trees are better.
>
> Note that finding tree size parameter only works after you set up a reliable validation scheme. Because you will find the tree size that maximizes your validation but that is meaningless if LB doesn't correlate with your validation.

## Cool links
- [Extensive EDA and Modeling XGB Hyperopt](https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt)
- [EDA and models](https://www.kaggle.com/artgor/eda-and-models) 
- [IEEE - LGB + Bayesian opt.](https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt)
- [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)