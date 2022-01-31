### - Import libraries - ###
import numpy as np     
import pandas as pd      
import seaborn as sns   
import matplotlib.pyplot as plt            
import sys


#from scipy import stats
from scipy.signal import periodogram

from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from xgboost import XGBRegressor

from myFunctions import myPrint



### - Import the data - ###

## Read in train dataset
df_train = pd.read_csv('train.csv', parse_dates=['date'], infer_datetime_format=True)
df_train['date'] = df_train.date.dt.to_period('D')

## Read in test dataset
df_test = pd.read_csv('test.csv', parse_dates=['date'])
df_test['date'] = df_test.date.dt.to_period('D')

## Read in holiday data set
holiday = pd.read_csv('holidays_events.csv', parse_dates=['date'])
holiday = holiday.set_index('date').to_period('D')
myPrint(holiday.isna().sum(), printmessage="Missing Holidays")

## Read in oil data set
oil = pd.read_csv('oil.csv', parse_dates=['date'])
oil = oil.set_index('date').to_period('d')

# Resample to fill in missing days. Fill NaN values with next value in series
oils = oil.resample('d').mean().fillna(method='bfill') 


## Create full data set for modifying features (separate before training the model!!)
full_df = pd.concat([df_train, df_test])
full_df.reset_index(drop=True, inplace=True)

## Set the index to be the store number, family and date labels
X_store = full_df.set_index(['store_nbr', 'family', 'date']).sort_index()
myPrint(X_store.head(), printmessage="Xstore head Data")

## Extract sales labels
y_ = X_store.loc[:,:,'2017':].dropna()['sales'].to_frame().unstack(['store_nbr', 'family'])
myPrint(y_, printmessage="labels")



### - Analysing the data and creating features - ###

### First look at correlation between store sales and the time

## Calculate average sales
av_sales = (X_store.groupby('date').mean().squeeze())['sales'].to_frame().dropna()
myPrint(av_sales, printmessage="averagesales")

## Plot periodogram of average sales as rough indicator of what Fourier components to use
fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
frequencies, spectrum = periodogram(av_sales.sales, fs=fs, detrend='linear', window="boxcar", scaling='spectrum')

_, ax = plt.subplots()
ax.step(frequencies, spectrum, color="purple")
ax.set_xscale("log")
ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
ax.set_xticklabels(["Annual (1)","Semiannual (2)","Quarterly (4)","Bimonthly (6)",
                    "Monthly (12)","Biweekly (26)","Weekly (52)","Semiweekly (104)"],rotation=30,)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylabel("Variance")
ax.set_title("Periodogram")
#plt.show()

## Find from periodogram that there is a monthly seasonality. 4 frequency components chosen so Fourier captures weekly changes. 
## Create monthly Fourer components
fourier = CalendarFourier(freq='M', order=4) 

## Also find there is a weekly dependence in sales and create dummies for the days of the week
## Could use Fourier for entire series, but would need 28 frequency components to model 
data_indices = X_store.loc[:,:,:].unstack(['store_nbr', 'family']).index
dp = DeterministicProcess(
    index=data_indices,
    constant=True,
    order=1, # Fit order
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X_full = dp.in_sample()
myPrint(X_full.columns.values, printmessage="Feature columns")


## Creat a new years day feature as we expect sales to drop 
X_nyd = X_full.copy()
X_nyd['new_year'] = (X_nyd.index.dayofyear ==1).astype('int') 


### Create pay day feature as we expect sales to increase CHECK DATA
X_pay = X_nyd.copy()

## Create a temporary days in month column
X_pay['day_in_m'] = X_pay.index.days_in_month 

## The pay days are every two weeks on the 15th and last day of month -- UNDERSTAND THIS
X_pay['pay_day'] = (X_pay.index.day == 16) | (X_pay.index.day == 1) | (X_pay.index.day == 14) | (X_pay.index.day == X_pay['day_in_m'] - 1) | (X_pay.index.day == 15) | (X_pay.index.day == X_pay['day_in_m'])

## Remove the days in month column as only needed for payday column
X_pay.drop(columns='day_in_m', inplace=True, axis=1)


### Create a national holiday feature as this will affect sales if the store is closed CHECK DATA

## Create a holiday dataframe
hol = holiday.loc['2017'].loc[holiday.loc['2017'].locale.isin(['National', 'Regional'])]

## Create a holiday column of 1s
hol['hol'] = 1      

## If holiday transferred, change to 0
hol.loc[(hol.type == 'Holiday') & (hol.transferred == True), 'hol'] = 0 

## Drop the other columns in holiday. Create dummies for other columns
hol = pd.get_dummies(hol.drop(columns=['locale','locale_name','description','transferred']), columns=['type'])

## Merge holiday dataframe with full dataframe
X_hol = X_pay.copy()
X_hol = pd.concat([X_hol, hol], axis=1).fillna(0)
X_hol.loc[X_hol.index.dayofweek.isin([5,6]), 'hol'] = 1


### Create an oil feature as oil price might also affect sales
X_oil = X_hol.copy()
X_oil['oil'] = oils.rolling(7).mean()



#### - Build the model - ####

### Split dataset into features and labels
## Don't train model on all years, only the 2017 data as most relevant for model
X_for_subm = X_oil.loc['2017':'2017-08-15'].dropna()
X_test = X_oil.loc['2017-08-16':].dropna()
#X_test = X_oil.loc['2017-08-16':'2017-08'].dropna()

myPrint(X_for_subm, printmessage="Final train data")
myPrint(y_, printmessage="Full feature dataset")


### MultiVariate Linear Regression Training ###

## Input X_for_subm has dimension n_dates X n_features and y_ has dimensions n_samples x n_targets 
## Here the targets are the products at each store e.g. target_1 = (BOOKs at Store_nbr 1), target_2 = (CARE at Store_nbr 1) etc.
model = LinearRegression()
model.fit(X_for_subm, y_) 

## Create predictions for train data set
y_pred = pd.DataFrame(model.predict(X_for_subm), columns=y_.columns, index=X_for_subm.index)
y_pred_ = y_pred.stack(['store_nbr', 'family'])
y_pred_.loc[y_pred_['sales'] < 0] = 0

## Plot the predictions against the actual data over 2017 ADD MORE PLOTS
plt.figure(figsize=(10,6));
plt.plot(y_.loc(axis=1)['sales',1, 'PRODUCE'].loc['2017'].values)
plt.plot(y_pred.loc(axis=1)['sales',1, 'PRODUCE'].loc['2017'].values)
#plt.show()

## Calculate Mean Absolute Error of model
mae = mean_absolute_error(y_pred.loc(axis=1)['sales',1, 'PRODUCE'].loc['2017'].values,y_.loc(axis=1)['sales',1, 'PRODUCE'].loc['2017'].values)
print("mae = ", mae)


### - Create submission file for simple Linear Regression model - ###
y_submit = pd.DataFrame(model.predict(X_test), columns=y_.columns, index=X_test.index)
y_submit_ = y_submit.stack(['store_nbr', 'family'])
y_submit_.loc[y_submit_['sales'] < 0] = 0

df_test_ = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()
y_submit_ = y_submit_.join(df_test_.id).reindex(columns=['id', 'sales'])
y_submit_.to_csv('submission_mvlr.csv', index=False)



### XGBoost training ###
## Regression good for trends and extrapolation, but the model is linear. It can't pick up nonlinear features/interactions
## or capture the error aspect in a MVLR. Therefore, we boost the model by fitting the residual noise.
## The other option is stacking where we train a model using the predictions as features.

### Create a residual. Need to move from stacked to unstacked form for the XGboost model as it can't handle multivariate data
y_res = y_.unstack().to_frame() - y_pred.unstack().to_frame()

## Create train and test dataset
store_sales = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()

X_xgb_old = store_sales.drop('sales', axis=1)
X_xgb = X_xgb_old.reset_index(['family', 'store_nbr'])

## Encode family and store number as features
le = LabelEncoder()  
X_xgb['family'] = le.fit_transform(X_xgb['family'])
X_xgb['store_nbr'] = le.fit_transform(X_xgb['store_nbr'])

## Create a day index and drop the id column
## Note we only use the day as a feature and not any other time features 
## as these are hopefully captures by the linear regression
X_xgb["day"] = X_xgb.index.day 
X_xgb.drop(columns=['id'], inplace=True)

## Again only analyse 2017
X_xgb = X_xgb.loc['2017']
X_xgb_old = X_xgb_old.loc[:,:,'2017']



### - Train the model - ###
model_2 = XGBRegressor()
model_2.fit(X_xgb, y_res)

## Create predictions for the training set
xgb_pred = pd.DataFrame(model_2.predict(X_xgb), index=X_xgb.index, columns=['sales'])

## Add the XGboost predicted values to the MVLR predictions
## N.B. We reset the index here so reinclude the store and family values 
y_boost = pd.DataFrame(y_pred.unstack().to_frame().values + xgb_pred.values, index=X_xgb_old.index, columns=['sales'])

## Set any negative sales to zero
y_boost.loc[y_boost.sales < 0] = 0
myPrint(y_boost, printmessage="Boosted prediction")

## Plot the new predicted sales vs the actual data
plt.figure(figsize=(20,6));
plt.plot(y_.loc(axis=1)['sales',1, 'PRODUCE'].loc['2017'].values)
plt.plot(y_boost.loc(axis=0)[1, 'PRODUCE'].loc['2017'].values)
#plt.show()

### Mean absolute error
mae_xgb = mean_absolute_error(y_boost.loc(axis=0)[1, 'PRODUCE'].loc['2017'].values, y_.loc(axis=1)['sales',1, 'PRODUCE'].loc['2017'].values)

## Compare mean absolute error. XGboost gives small improvement
## However, find on submission to Kaggle the model performs much worse - seems to overfit maybe?
print("mae = ", mae, " ", mae_xgb)



### - Predictions for test set - ###
store_sales_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()
X_xgb_old_test = store_sales_test
X_xgb_test = X_xgb_old_test.reset_index(['family', 'store_nbr'])
X_xgb_test['family'] = le.fit_transform(X_xgb_test['family'])
X_xgb_test['store_nbr'] = le.fit_transform(X_xgb_test['store_nbr'])
X_xgb_test["day"] = X_xgb_test.index.day 
X_xgb_test.drop(columns=['id'], inplace=True)
myPrint(X_xgb_test)


xgb_pred_test = pd.DataFrame(model_2.predict(X_xgb_test), index=X_xgb_test.index, columns=['sales'])
y_boost_test = pd.DataFrame(y_submit.unstack().to_frame().values + xgb_pred_test.values, index=X_xgb_old_test.index, columns=['sales'])
y_boost_test.loc[y_boost_test.sales < 0] = 0
myPrint(y_boost_test, printmessage="Test boosted prediction")

## Add test predictions to old test predictions
y_submit_ = y_boost_test.join(df_test_.id).reindex(columns=['id', 'sales'])
y_submit_.to_csv('submission_xgb.csv', index=False)

