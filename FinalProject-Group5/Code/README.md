1. Project Description:

NOAA (National Oceanic and Atmospheric Administration) records the occurrence of storms and other significant weather phenomena having sufficient intensity to cause loss of life, injuries
, significant property damage, and/or disruption to commerce.

NOAA stores the observations of storm events in a database of csv files (https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/). We are using the features and observations from this 
data to predict the property damage caused by any of the storm events in United States.

2. Project Status: Complete

3. Usage Guidance: Please execute the code in following order

3.1 Pipeline.py - 

This file calls two important files: PreProcessing.py and Model.py.

PreProcessing.py does data extraction and all important data pre-processing steps.

Model.py uses the processed data and trains and predicts the model.

3.2 Main.py - 

This file will use the trained models from Pipeline.py and, using PyQt5, build dashboard to show EDA and Model results.

4. Technology Used: Python
