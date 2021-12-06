# import PreProcessing
# import Model
#
# PreProcessing.main()
# Model.main()

# Generic Libraries
import pickle

import numpy as np
import pandas as pd

# Preprocessing
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold



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

# Create train test split
X = df_train.loc[:, ~df_train.columns.isin(['TOTAL_DAMAGE', 'YEAR'])]

Y = df_train['TOTAL_DAMAGE']

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


model = RandomForestRegressor()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, verbose=3)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
