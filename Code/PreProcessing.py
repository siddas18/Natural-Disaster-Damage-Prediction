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


def main():
    # ------------- Data Extraction ---------------

    def get_NOAA_data():
        URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
        r = requests.get(URL)
        file_names = pd.read_html(r.text)[0]['Name']
        events_file_names = file_names[file_names.str.contains("details", na=False)]
        noaa_list = []

        print("Extracting CSV files...")
        for file in events_file_names:
            full_URL = URL + file
            with gzip.open(requests.get(full_URL, stream=True).raw) as f:
                noaa_list.append(pd.read_csv(f))

        df = pd.concat(noaa_list)

        print("Completed")
        return df

    def pickle_source_data():
        noaa_source_df = get_NOAA_data()
        home_dir = os.getcwd()
        data_dir = os.path.join(home_dir, "Data")
        try:
            os.mkdir(data_dir)
            os.chdir(data_dir)
        except OSError:
            os.chdir(data_dir)
            for file in os.listdir():
                os.remove(file)
        noaa_source_df.to_pickle('noaa_source_data.pkl')
        os.chdir(home_dir)
        return noaa_source_df

    NOAA_df = pickle_source_data()

    def get_EPA_data():
        ground_temp_df = pd.read_csv('https://www.epa.gov/sites/default/files/2021-04/temperature_fig-1.csv',
                                     skiprows=6,
                                     usecols=[0, 1], encoding='latin1')
        ground_temp_df.columns = ["YEAR", "SURFACE_TEMP_DIFF"]

        greenhouse_df = pd.read_csv("https://www.epa.gov/sites/default/files/2021-04/us-ghg-emissions_fig-1.csv",
                                    skiprows=6)
        greenhouse_df.columns = ["YEAR", "CO2", "METHANE", "N2O", "HFC_PFC_SF6_NF3"]

        precipitation_df = pd.read_csv('https://www.epa.gov/sites/default/files/2021-04/heavy-precip_fig-1.csv',
                                       skiprows=6)
        precipitation_df.columns = ["YEAR", "PPT_LAND_AREA", "PPT_NINE_YEAR_AVG"]

        sea_level_df = pd.read_csv('https://www.epa.gov/sites/default/files/2021-04/sea-level_fig-1.csv', skiprows=6)
        sea_level_df.columns = ["YEAR", "CSIRO_ADJ_SEA_LEVEL", "CSIRO_LOWER", "CSIRO_UPPER", "NOAA_ADJ_SEA_LEVEL"]
        sea_level_df.loc[sea_level_df['YEAR'] > 1992, 'CSIRO_ADJ_SEA_LEVEL'] = sea_level_df.loc[
            sea_level_df['YEAR'] > 1992, 'NOAA_ADJ_SEA_LEVEL']
        sea_level_df.drop(['CSIRO_LOWER', 'CSIRO_UPPER', 'NOAA_ADJ_SEA_LEVEL'], axis=1, inplace=True)

        seasonal_temp_df = pd.read_csv('https://www.epa.gov/sites/default/files/2021-04/seasonal-temperature_fig-1.csv',
                                       skiprows=6)
        seasonal_temp_df.columns = ["YEAR", "WINTER_ANOMALY", "SPRING_ANOMALY", "SUMMER_ANOMALY", "FALL_ANOMALY"]

        arctic_ice_df = pd.read_csv('https://www.epa.gov/sites/default/files/2021-03/arctic-sea-ice_fig-1.csv',
                                    skiprows=6)
        arctic_ice_df.columns = ["YEAR", "ICE_CVG_MARCH", "ICE_CVG_SEP"]

        glacier_df = pd.read_csv('https://www.epa.gov/sites/default/files/2021-03/glaciers_fig-1.csv', skiprows=6)
        glacier_df.columns = ["YEAR", "GLACIER_MASS_BAL", "NUM_OBS"]
        glacier_df.drop(['NUM_OBS'], axis=1, inplace=True)

        dfs = [ground_temp_df, greenhouse_df, precipitation_df, sea_level_df, seasonal_temp_df, arctic_ice_df,
               glacier_df]
        epa_df = reduce(lambda left, right: pd.merge(left, right, how="outer", on="YEAR"), dfs)
        epa_df = epa_df[epa_df.YEAR >= 1950]

        return epa_df

    epa_source_df = get_EPA_data()

    # ------------- Reading the source file in a dataframe ---------------

    NOAA_df = pd.read_pickle('Data/noaa_source_data.pkl')

    # ------------- Pre-Processing ---------------

    # Update DAMAGE_PROPERTY and DAMAGE_CROPS variable

    def replace_str2num(x):
        if type(x) == float or type(x) == int:
            return float(x)
        num = 1 if x[:-1] == '' else x[:-1]
        if x[-1] == 'T':
            return float(num) * 1000000000000
        elif x[-1] == 'B':
            return float(num) * 1000000000
        elif x[-1] == 'M':
            return float(num) * 1000000
        elif x[-1] == 'K' or x[-1] == 'k':
            return float(num) * 1000
        elif x[-1] == 'h' or x[-1] == 'H':
            return float(num) * 100
        elif x[-1] == '?':
            return float(num)
        else:
            return float(x)

    # Split the MAGNITUDE variable into WIND and HAIL

    def winds(x):
        if x['MAGNITUDE_TYPE'] in ['EG', 'E', 'M', 'ES', 'MG', 'MS']:
            return x['MAGNITUDE']

    def hail(x):
        if x['MAGNITUDE_TYPE'] not in ['EG', 'E', 'M', 'ES', 'MG', 'MS']:
            return x['MAGNITUDE']

    # Swap Missing Data variables

    def missing_swap(df, col1, col2):
        df.loc[~df[col1].isnull() & df[col2].isnull(), col2] = df.loc[~df[col1].isnull() & df[col2].isnull(), col1]
        df.loc[df[col1].isnull() & ~df[col2].isnull(), col1] = df.loc[df[col1].isnull() & ~df[col2].isnull(), col2]
        return df

    # Rename EVENT_TYPE values

    rename_event_dict = {
        'TORNADOES, TSTM WIND, HAIL': 'Tornadoes, Thunderstorm Wind, Hail',
        'THUNDERSTORM WINDS LIGHTNING': 'Thunderstorm Wind, Lightning',
        'THUNDERSTORM WINDS/ FLOOD': 'Thunderstorm Wind, Flood',
        'THUNDERSTORM WINDS/FLOODING': 'Thunderstorm Wind, Flood',
        'THUNDERSTORM WIND/ TREES': 'Thunderstorm Wind, Trees',
        'THUNDERSTORM WIND/ TREE': 'Thunderstorm Wind, Trees',
        'THUNDERSTORM WINDS/HEAVY RAIN': 'Thunderstorm Wind, Heavy Rain',
        'TORNADO/WATERSPOUT': 'Tornado, Waterspout',
        'THUNDERSTORM WINDS FUNNEL CLOU': 'Thunderstorm Wind, Funnel Cloud',
        'THUNDERSTORM WINDS/FLASH FLOOD': 'Thunderstorm Wind, Flash Flood',
        'HAIL/ICY ROADS': 'Hail, Icy Roads',
        'HAIL FLOODING': 'Hail, Flood',
        'THUNDERSTORM WINDS HEAVY RAIN': 'Thunderstorm Wind, Heavy Rain',
        'Hurricane (Typhoon)': 'Hurricane'
    }

    # Fix the timezone format

    timezone_dict = {
        'GST': ['GST10'],
        'AST': ['AST-4', 'AST'],
        'EST': ['EST', 'EST-5', 'ESt', 'EDT'],
        'CST': ['CST', 'CST-6', 'CSt', 'CSC', 'SCT', 'GMT', 'UNK', 'CDT'],
        'MST': ['MST', 'MST-7', 'MDT'],
        'PST': ['PST', 'PST-8', 'PDT'],
        'AKST': ['AKST-9'],
        'HST': ['HST-10', 'HST'],
        'SST': ['SST-11', 'SST']
    }

    def timezone_mapping(x):
        for key, val in timezone_dict.items():
            if x in val:
                return key

    # Fix azimuth mapping

    azimuth_mapping = {'N/A': ['ND', 'EE', 'TO', 'MI', 'M', 'EST', 'EAS', 'TH', 'WES']}

    def dict_mapping(x):
        for key, val in timezone_dict.items():
            if x in val:
                return key

    # Using data after Year 1950 and checking how many columns have NULL values

    NOAA_df = NOAA_df[NOAA_df['YEAR'] > 1950]

    # Damage variables cleaning

    NOAA_df['DAMAGE_PROPERTY'] = NOAA_df.DAMAGE_PROPERTY.map(replace_str2num)

    NOAA_df['DAMAGE_CROPS'] = NOAA_df.DAMAGE_CROPS.map(replace_str2num)

    # Removing inconsistencies
    NOAA_df = missing_swap(NOAA_df, 'BEGIN_RANGE', 'END_RANGE')
    NOAA_df = missing_swap(NOAA_df, 'BEGIN_LAT', 'END_LAT')
    NOAA_df = missing_swap(NOAA_df, 'BEGIN_LON', 'END_LON')
    NOAA_df = missing_swap(NOAA_df, 'BEGIN_AZIMUTH', 'END_AZIMUTH')
    NOAA_df = missing_swap(NOAA_df, 'BEGIN_LOCATION', 'END_LOCATION')

    # calculating distance w.r.t Latitude and Longitude

    def geo_distance(x):
        # Source : https://en.wikipedia.org/wiki/Haversine_formula
        p = pi / 180
        lat1 = x['BEGIN_LAT']
        lat2 = x['END_LAT']
        lon1 = x['BEGIN_LON']
        lon2 = x['END_LON']
        R = 6371
        dLat = p * (lat2 - lat1)
        dLon = p * (lon2 - lon1)
        a = sin(dLat / 2) * 2 + cos(p * lat1) * cos(p * lat2) * sin(dLon / 2) * 2
        if a < 0:
            return a
        else:
            return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    NOAA_df['GEO_DISTANCE'] = NOAA_df.apply(lambda x: geo_distance(x), axis=1)

    # calculate duration of the storm event

    def calc_duration(x):
        begin_dt = dt.strptime(x['BEGIN_DATE_TIME'], "%d-%b-%y %H:%M:%S")
        end_dt = dt.strptime(x['END_DATE_TIME'], "%d-%b-%y %H:%M:%S")
        difference = end_dt - begin_dt
        difference_days = difference.days + difference.seconds / 86400
        return difference_days

    NOAA_df['DURATION_OF_STORM'] = NOAA_df.apply(lambda x: calc_duration(x), axis=1)

    # Update EVENT_TYPE column

    NOAA_df.replace(
        {'EVENT_TYPE': {'THUNDERSTORMWIND/TREE': 'ThunderstormWind', 'THUNDERSTORMWIND/TREES': 'ThunderstormWind'
            , 'THUNDERSTORMWINDS/FLASHFLOOD': 'ThunderstormWind', 'THUNDERSTORMWINDS/FLOODING': 'ThunderstormWind'
            , 'THUNDERSTORMWINDS/HEAVYRAIN': 'ThunderstormWind', 'THUNDERSTORMWINDSFUNNELCLOU': 'ThunderstormWind'
            , 'THUNDERSTORMWINDSHEAVYRAIN': 'ThunderstormWind', 'THUNDERSTORMWINDSLIGHTNING': 'ThunderstormWind'
            , 'HAIL/ICYROADS': 'Hail', 'HAILFLOODING': 'Hail', 'Hurricane(Typhoon)': 'Hurricane'}})

    # Dropping NAN values - if required

    NOAA_df.dropna(subset=["DAMAGE_PROPERTY"], inplace=True)
    NOAA_df.dropna(subset=["DAMAGE_CROPS"], inplace=True)

    # Impute variable values

    def impute_NOAA_data(df):
        drop_list = ['EVENT_NARRATIVE', 'EPISODE_NARRATIVE', 'EPISODE_ID', 'MAGNITUDE', 'BEGIN_LAT', 'END_LAT',
                     'BEGIN_LON',
                     'END_LON'
            , 'BEGIN_DATE_TIME', 'END_DATE_TIME', 'STATE_FIPS', 'TOR_OTHER_CZ_FIPS', 'WFO', 'SOURCE', 'CATEGORY',
                     'CZ_FIPS',
                     'DATA_SOURCE'
            , 'TOR_OTHER_WFO', 'EVENT_ID', 'BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'END_YEARMONTH', 'END_DAY',
                     'END_TIME']

        impute_mean_list = ['DEATHS_DIRECT', 'DEATHS_INDIRECT', 'INJURIES_DIRECT', 'INJURIES_INDIRECT']

        # imputing damage columns with 0 for the time-being
        impute_zero_list = ['BEGIN_RANGE', 'END_RANGE', 'WIND_SPEED', 'HAIL_SIZE', 'GEO_DISTANCE', 'TOR_LENGTH',
                            'TOR_WIDTH']

        impute_NA_list = ['CZ_NAME', 'STATE', 'MAGNITUDE_TYPE', 'BEGIN_AZIMUTH', 'END_AZIMUTH', 'BEGIN_LOCATION',
                          'END_LOCATION', 'FLOOD_CAUSE', 'TOR_F_SCALE'
            , 'TOR_OTHER_CZ_STATE', 'TOR_OTHER_CZ_NAME']

        # Splitting magnitude variable into constituent attributes
        df['WIND_SPEED'] = df.apply(winds, axis=1)
        df['HAIL_SIZE'] = df.apply(hail, axis=1)

        df['EVENT_TYPE'] = df['EVENT_TYPE'].apply(
            lambda x: rename_event_dict[x] if rename_event_dict.get(x) != None else x)
        df['COLD_WEATHER_EVENT'] = df['EVENT_TYPE'].str.contains(
            'Hail|Winter|Snow|Chill|Cold|Frost|Freeze|Blizzard|Ice|Avalanche').map({True: 1, False: 0})
        df['WINDY_EVENT'] = df['EVENT_TYPE'].str.contains('Wind|Tornado|Thunderstorm|Cloud|Storm').map(
            {True: 1, False: 0})
        df['WATER_EVENT'] = df['EVENT_TYPE'].str.contains(
            'Flood|Marine|Rain|Hurricane|Tide|Lake|Seiche|Tsunami|Sleet|Water').map({True: 1, False: 0})
        df.loc[:, 'CZ_TIMEZONE'] = df.loc[:, 'CZ_TIMEZONE'].apply(lambda x: timezone_mapping(x))
        df.loc[:, 'BEGIN_AZIMUTH'] = df.loc[:, 'BEGIN_AZIMUTH'].str.upper().apply(
            lambda x: dict_mapping(x) if dict_mapping(x) != None else x)
        df.loc[:, 'END_AZIMUTH'] = df.loc[:, 'END_AZIMUTH'].str.upper().apply(
            lambda x: dict_mapping(x) if dict_mapping(x) != None else x)

        # Imputing string columns with missing values with NA
        for col in impute_NA_list:
            df[col] = df[col].astype('str').apply(lambda x: 'NA' if x == 'nan' else x)  # changed from N/A to NA

        # Imputing float columns having missing values with 0.0
        for col in impute_zero_list:
            df[col] = df[col].fillna(0.0)

        # Imputing latitude and longitudes with average value
        # for col in impute_mean_list:
        #     df[col] = df[col].fillna(np.mean)

        # Dropping text and ID columns
        for col in drop_list:
            df.drop(col, axis=1, inplace=True)

        return df

    imputed_NOAA_df = impute_NOAA_data(NOAA_df.copy())
    imputed_NOAA_df.head()

    # EPA Data processing

    def impute_EPA_DATA(df, breaks):
        fillable_cols = df.columns[df.isnull().sum() > 0]
        for col in fillable_cols:
            temp_df = df[['YEAR', col]]
            present_df = temp_df[~ temp_df[col].isnull()]
            null_df = temp_df[temp_df[col].isnull()]
            years = sorted(np.random.choice(present_df['YEAR'], breaks))
            input_df = present_df[present_df['YEAR'].isin(years)]
            func = scipy.interpolate.interp1d(input_df['YEAR'], input_df[col], fill_value="extrapolate")
            temp_df['INTERPOLATION'] = func(temp_df['YEAR'])
            df[col] = temp_df.apply(lambda x: x['INTERPOLATION'] if isnan(x[col]) else x[col], axis=1)
        return df

    imputed_EPA_df = impute_EPA_DATA(epa_source_df.copy(), 6)
    imputed_EPA_df.head()

    # Reading the cleaned data into a pickle file

    NOAA = imputed_NOAA_df

    df_train = NOAA[(NOAA["YEAR"] > 2005)]

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


if __name__ == '__main__':
    main()
