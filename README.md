# Kaggle_Store_sales
Repository for the notebook (.ipynb) and source code (.py) used in the Kaggle competition "Store sales - Time series forecasting".

Involved implementing multivariate linear regression (MVLR) to predict the future sales across 54 stores selling 33
product types in Ecuador across 2013 to 2017.
After applying MVLR, a second model (XGboost) was trained on the residuals to try and improve the model 
score, and just for practice.
A submission file for the predictions generated with MVLR and with MVLR and XGboost was generated. The competition score was the root mean squared log error (rmsle).
For the MVLR model alone, the rmsle was 0.5. For MVLR+XGboost model, the rmsle was 1.03.

Adding in the XGboost unfortunately made the model perform worse, and further analysis is needed into why this is the case. The XGboost model is trained on the on-promotion feature of the dataset, as well as the store and product features, used to differentiate the different time series. It is possible these features are actually mixing the data between products and stores, which would give errors. It could also be that better features, such as lag features, need to be added to the model. Finally, it could also be that XGboost just isn't suitable and the time series can't be approximated much better than with just linear regression.
