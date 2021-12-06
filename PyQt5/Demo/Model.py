# Generic Libraries
import pickle
import pandas as pd

# Preprocessing
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.catch_warnings()
warnings.simplefilter("ignore")

df_train = pd.read_pickle('Data/cleaned_NAN_removed.pkl')


# GET DUMMIES and assign labels to categorical columns

def mapping(xx):
    dict = {}
    count = -1
    for x in xx:
        dict[x] = count + 1
        count = count + 1
    return dict


for i in ['CZ_NAME', 'BEGIN_LOCATION', 'END_LOCATION', 'TOR_OTHER_CZ_STATE', 'TOR_OTHER_CZ_NAME']:
    unique_tag = df_train[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    df_train[i] = df_train[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

df_train = pd.get_dummies(df_train,
                          prefix=['STATE', 'MONTH_NAME', 'EVENT_TYPE', 'CZ_TYPE', 'CZ_TIMEZONE', 'BEGIN_AZIMUTH',
                                  'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'TOR_F_SCALE', 'END_AZIMUTH']
                          , columns=['STATE', 'MONTH_NAME', 'EVENT_TYPE', 'CZ_TYPE', 'CZ_TIMEZONE', 'BEGIN_AZIMUTH',
                                     'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'TOR_F_SCALE', 'END_AZIMUTH'])
df_train.head()

df_train.to_pickle('Data/df_train.pkl')

# Create train test split
X = df_train.loc[:, ~df_train.columns.isin(['TOTAL_DAMAGE', 'YEAR'])]

Y = df_train['TOTAL_DAMAGE']

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Model 1 - Linear Regression

reg_lr = LinearRegression()
reg_lr.fit(X_train, y_train)

# make predictions for test data
y_pred = reg_lr.predict(X_test)

MSE = mean_squared_error(y_pred, y_test)
print("Mean Square Value", MSE)

print("Training R-square value", reg_lr.score(X_train, y_train))

print("R-Square Value", r2_score(y_test, y_pred))

pickle.dump(reg_lr, open('Data/LR_Model.pkl', 'wb'))

# Model 2 - Random Forest

reg_rf = RandomForestRegressor(n_estimators=100, oob_score='TRUE', n_jobs=-1, random_state=50, max_features="auto",
                               min_samples_leaf=50)

# perform training
reg_rf.fit(X_train, y_train)

# make predictions

# prediction on test using all features
y_pred = reg_rf.predict(X_test)

MSE = mean_squared_error(y_pred, y_test)
print("Mean Square Value", MSE)

print("Training R-square value", reg_rf.score(X_train, y_train))

print("R-Square Value", r2_score(y_test, y_pred))

pickle.dump(reg_rf, open('Data/RF_Model.pkl', 'wb'))

# Model 3 - XGBoost

xgb = XGBRegressor(learning_rate=0.01, subsample=0.7, max_depth=5, n_estimators=100, colsample_bytree=0.8)
xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)])

# make predictions for test data
y_pred = xgb.predict(X_test)
y_pred_score = [round(value) for value in y_pred]

MSE = mean_squared_error(y_pred, y_test)
print("Mean Square Value xb", MSE)

print("Training R-square value xb", xgb.score(X_train, y_train))

print("R-Square Value xb", r2_score(y_test, y_pred))

pickle.dump(xgb, open('Data/XGB_Model.pkl', 'wb'))

# Model 4 - Ensemble Model

# create a dictionary of our models
estimators = [('LR', reg_lr), ('RF', reg_rf), ('XGB', xgb)]

# create our voting classifier, inputting our models
ensemble = VotingRegressor(estimators)

# fit model to training data
ensemble.fit(X_train, y_train)
# test our model on the test data

y_pred = ensemble.predict(X_test)

print("R-Square Value", r2_score(y_test, y_pred))

pickle.dump(ensemble, open('Data/Ensemble_Model.pkl', 'wb'))