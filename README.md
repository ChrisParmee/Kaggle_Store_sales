# Kaggle_Store_sales
Repository for the notebook (.ipynb) and source code (.py) used in the Kaggle competition "Store sales - Time series forecasting".

Involved implementing multivariate linear regression (MVLR) to predict the future sales across 54 stores selling 33
product types in Ecuador across 2013 to 2017.
After applying MVLR, a second model (XGboost) was trained on the residuals to improve the model 
score.
A submission file for the predictions generated with MVLR and with MVLR and XGboost was generated.
The submission score for just MVLR was __
The submission score for MVLR+XGboost was __.

Adding in the XGboost made the model marginally better on the training data, but clearly resulting in a worse result on submission, which seems to indicate overfitting.
However, I would need to carry out more analysis of the model and data to confirm this. 
