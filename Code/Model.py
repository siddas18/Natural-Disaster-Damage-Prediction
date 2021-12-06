# Generic Libraries
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def main():
    df_train = pd.read_pickle('Data/cleaned_NAN_removed.pkl')

    # GET DUMMIES and assign labels to categorical columns

    def mapping(xx):
        dict = {}
        count = -1
        for x in xx:
            dict[x] = count + 1
            count = count + 1
        return dict

    def tor_scale(x):
        if type(x) == float or x[-1] == 'U' or x[-1] == 'A':
            return 0
        else:
            return int(x[-1]) + 1

    df_train.loc[:, 'TOR_F_SCALE'] = df_train.loc[:, 'TOR_F_SCALE'].str.upper().apply(lambda x: tor_scale(x))

    for i in ['CZ_NAME', 'BEGIN_LOCATION', 'END_LOCATION', 'TOR_OTHER_CZ_STATE', 'TOR_OTHER_CZ_NAME']:
        unique_tag = df_train[i].value_counts().keys().values
        dict_mapping = mapping(unique_tag)
        df_train[i] = df_train[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

    df_train = pd.get_dummies(df_train,
                              prefix=['STATE', 'MONTH_NAME', 'EVENT_TYPE', 'CZ_TYPE', 'CZ_TIMEZONE', 'BEGIN_AZIMUTH',
                                      'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'END_AZIMUTH']  # 'TOR_F_SCALE',
                              , columns=['STATE', 'MONTH_NAME', 'EVENT_TYPE', 'CZ_TYPE', 'CZ_TIMEZONE', 'BEGIN_AZIMUTH',
                                         'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'END_AZIMUTH'])  # 'TOR_F_SCALE',

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
    print("LR Mean Square Value", MSE)

    print("LR Training R-square value", reg_lr.score(X_train, y_train))

    print("LR R-Square Value", r2_score(y_test, y_pred))

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
    print("RF Mean Square Value", MSE)

    print("RF Training R-square value", reg_rf.score(X_train, y_train))

    print("RF R-Square Value", r2_score(y_test, y_pred))

    pickle.dump(reg_rf, open('Data/RF_Model.pkl', 'wb'))

    # Model 3 - XGBoost

    xgb = XGBRegressor(learning_rate=0.01, subsample=0.7, max_depth=5, n_estimators=500, colsample_bytree=0.8)
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)])

    # make predictions for test data
    y_pred = xgb.predict(X_test)
    y_pred_score = [round(value) for value in y_pred]

    MSE = mean_squared_error(y_pred, y_test)
    print("XGB Mean Square Value", MSE)

    print("XGB Training R-square value", xgb.score(X_train, y_train))

    print("XGB R-Square Value", r2_score(y_test, y_pred))

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

    MSE = mean_squared_error(y_pred, y_test)
    print("Ensemble Mean Square Value", MSE)

    print("Ensemble Training R-square value", xgb.score(X_train, y_train))

    print("Ensemble R-Square Value", r2_score(y_test, y_pred))

    pickle.dump(ensemble, open('Data/Ensemble_Model.pkl', 'wb'))

    pred1 = reg_lr.predict(X_test[50:])
    pred2 = reg_rf.predict(X_test[50:])
    pred3 = xgb.predict(X_test[50:])
    pred4 = ensemble.predict(X_test[50:])

    plt.figure()
    plt.plot(pred1, "gd", label="GradientBoostingRegressor")
    plt.plot(pred2, "b^", label="RandomForestRegressor")
    plt.plot(pred3, "ys", label="LinearRegression")
    plt.plot(pred4, "r*", ms=10, label="VotingRegressor")

    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.ylabel("predicted")
    plt.xlabel("training samples")
    plt.legend(loc="best")
    plt.title("Regressor predictions and their average")

    plt.show()

    def plot_feature_importance(importance, names, model_type):

        # Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        # Create a DataFrame using a Dictionary
        df = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(df)

        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

        fi_df = fi_df[:10]
        # Define size of bar plot
        plt.figure(figsize=(15, 12))
        # Plot Seaborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        # Add chart labels
        plt.title(model_type + 'FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        plt.show()

    plot_feature_importance(reg_rf.feature_importances_, X_train.columns, 'RANDOM FOREST')
    plot_feature_importance(xgb.feature_importances_, X_train.columns, 'XGBoost')


if __name__ == '__main__':
    main()
