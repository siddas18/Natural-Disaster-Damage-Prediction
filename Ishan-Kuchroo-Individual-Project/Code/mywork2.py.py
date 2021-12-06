# Generic Libraries
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
from math import sqrt, isnan, pi, sin, cos, atan2
import requests
import gzip
from functools import reduce
import scipy.interpolate

import warnings

warnings.catch_warnings()
warnings.simplefilter("ignore")

df_train = pd.read_pickle('Data/cleaned_NAN_removed.pkl')

# Removing Outliers

print("Old Shape: ", df_train.shape)

Quart1 = df_train.quantile(0.25)
Quart3 = df_train.quantile(0.75)
Range = Quart3 - Quart1

df_train = df_train[~((df_train < (Quart1 - 1.5 * Range)) | (df_train > (Quart3 + 1.5 * Range))).any(axis=1)]

print("New Shape: ", df_train.shape)

df_train['TOTAL_DAMAGE'] = df_train['DAMAGE_PROPERTY'] + df_train['DAMAGE_CROPS']

df_train.drop('DAMAGE_PROPERTY', axis=1, inplace=True)
df_train.drop('DAMAGE_CROPS', axis=1, inplace=True)

df_train.to_pickle('Data/cleaned_NAN_removed.pkl')

