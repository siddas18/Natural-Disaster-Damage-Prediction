##################################################
### Created by Lilian Sao de Rivera
### Project Name : The economics of happiness
### Date 04/23/2017
### Data Mining
##################################################

import sys

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
# import pyqtgraph as pg

from scipy import interpolate
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error, r2_score

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.btnExecute,5,0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)


        self.lblMSE = QLabel('Mean Square Value:')
        self.txtMSE = QLineEdit()

        self.lblTRV = QLabel('Training R-square value:')
        self.txtTRV = QLineEdit()

        self.lblTSV = QLabel('TR-Square Value:')
        self.txtTSV = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblMSE)
        self.groupBox2Layout.addWidget(self.txtMSE)

        self.groupBox2Layout.addWidget(self.lblTRV)
        self.groupBox2Layout.addWidget(self.txtTRV)

        self.groupBox2Layout.addWidget(self.lblTSV)
        self.groupBox2Layout.addWidget(self.txtTSV)


        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        # self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)
        # self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,0,1)
        # self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''


        # Assign the X and y to run the Random Forest Classifier

        X_dt = df_train.loc[:, ~df_train.columns.isin(['TOTAL_DAMAGE', 'YEAR'])]
        y_dt = df_train['TOTAL_DAMAGE']

        class_le = LabelEncoder()

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.3, random_state=100)


        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = loaded_model_rf.predict(X_test)


        self.ff_mse = mean_squared_error(y_pred,y_test)
        self.txtMSE.setText(str(self.ff_mse))

        self.ff_trv = (loaded_model_rf.score(X_train, y_train)*100)
        self.txtTRV.setText(str(self.ff_trv))

        self.ff_tsv = (r2_score(y_test, y_pred) * 100)
        self.txtTSV.setText(str(self.ff_tsv))


        ######################################
        # Graph - 3 Feature Importances
        #####################################

        feature_importance = np.array(loaded_model_rf.feature_importances_)
        feature_names = np.array(X_train.columns)

        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)

        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

        X_Features = fi_df.feature_importance[:15]
        y_Importance = fi_df.feature_names[:15]

        self.ax3.barh(y_Importance,X_Features )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()


class XGBoost(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(XGBoost, self).__init__()

        self.Title ="XGBoost"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('XGBoost')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.btnExecute = QPushButton("Execute XGBoost")
        self.btnExecute.clicked.connect(self.update)


        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.btnExecute,6,0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblMSE = QLabel('Mean Square Value:')
        self.txtMSE = QLineEdit()

        self.lblTRV = QLabel('Training R-square value:')
        self.txtTRV = QLineEdit()

        self.lblTSV = QLabel('TR-Square Value:')
        self.txtTSV = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblMSE)
        self.groupBox2Layout.addWidget(self.txtMSE)

        self.groupBox2Layout.addWidget(self.lblTRV)
        self.groupBox2Layout.addWidget(self.txtTRV)

        self.groupBox2Layout.addWidget(self.lblTSV)
        self.groupBox2Layout.addWidget(self.txtTSV)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        # self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2, 0, 2)
        # self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3, 0, 1)
        # self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):
        '''
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        # We process the parameters

        # Assign the X and y to run the Random Forest Classifier

        X_dt = df_train.loc[:, ~df_train.columns.isin(['TOTAL_DAMAGE', 'YEAR'])]
        y_dt = df_train['TOTAL_DAMAGE']

        class_le = LabelEncoder()

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.3, random_state=100)

        y_pred = loaded_model_xgb.predict(X_test)

        self.ff_mse = mean_squared_error(y_pred, y_test)
        self.txtMSE.setText(str(self.ff_mse))

        self.ff_trv = (loaded_model_xgb.score(X_train, y_train) * 100)
        self.txtTRV.setText(str(self.ff_trv))

        self.ff_tsv = (r2_score(y_test, y_pred) * 100)
        self.txtTSV.setText(str(self.ff_tsv))

        ######################################
        # Graph - 3 Feature Importances
        #####################################

        feature_importance = np.array(loaded_model_xgb.feature_importances_)
        feature_names = np.array(X_train.columns)

        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)

        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

        X_Features = fi_df.feature_importance[:15]
        y_Importance = fi_df.feature_names[:15]

        self.ax3.barh(y_Importance, X_Features)
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()



class CorrelationPlot(QMainWindow):
    #;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a correlation plot
    # It presents all the features plus the happiness score
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-----------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        #::--------------------------------------------------------
        super(CorrelationPlot, self).__init__()

        self.Title = 'Correlation Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Correlation Plot Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)


        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[2], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,0,2)
        self.groupBox1Layout.addWidget(self.feature3,0,3)
        self.groupBox1Layout.addWidget(self.feature4,1,0)
        self.groupBox1Layout.addWidget(self.feature5,1,1)
        self.groupBox1Layout.addWidget(self.feature6,1,2)
        self.groupBox1Layout.addWidget(self.feature7,1,3)
        self.groupBox1Layout.addWidget(self.feature8,2,0)
        self.groupBox1Layout.addWidget(self.feature9,2,1)
        self.groupBox1Layout.addWidget(self.btnExecute,2,2)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()


        self.groupBox2 = QGroupBox('Correlation Plot')
        self.groupBox2Layout= QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox2Layout.addWidget(self.canvas)


        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(900, 700)
        self.show()
        self.update()

    def update(self):

        #::------------------------------------------------------------
        # Populates the elements in the canvas using the values
        # chosen as parameters for the correlation plot
        #::------------------------------------------------------------
        self.ax1.clear()

        ff_noaa_cor = ff_noaa[features_list]

        def mapping(xx):

            dict = {}
            count = -1
            for x in xx:
                dict[x] = count + 1
                count = count + 1
            return dict

        for i in ["MAGNITUDE_TYPE", "EVENT_TYPE", "STATE","CZ_TIMEZONE","WINDY_EVENT","YEAR","MONTH_NAME",'CZ_NAME']:
            unique_tag = ff_noaa_cor[i].value_counts().keys().values
            dict_mapping = mapping(unique_tag)
            ff_noaa_cor[i] = ff_noaa_cor[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

        X_1 = ff_noaa["TOTAL_DAMAGE"]

        list_corr_features = pd.DataFrame(ff_noaa["TOTAL_DAMAGE"])
        if self.feature0.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[7]]],axis=1)

        if self.feature8.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[8]]],axis=1)

        if self.feature9.isChecked():
            list_corr_features = pd.concat([list_corr_features, ff_noaa_cor[features_list[9]]], axis=1)


        vsticks = ["dummy"]
        vsticks1 = list(list_corr_features.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = list_corr_features.corr(method ='kendall')
        self.ax1.matshow(res_corr, cmap= plt.cm.get_cmap('Blues', 14), interpolation='nearest')
        self.ax1.set_xticks(np.arange(len(vsticks1[1:])))
        self.ax1.set_yticks(np.arange(len(vsticks1[1:])))
        self.ax1.set_yticklabels(vsticks1[1:])
        self.ax1.set_xticklabels(vsticks1[1:],rotation = 90)
        self.fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        self.fig.canvas.draw_idle()


class DPGraphs(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the happiness score
    # methods
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a dotplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(DPGraphs, self).__init__()

        self.Title = "Features vrs TOTAL_DAMAGE"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["MAGNITUDE_TYPE", "WIND_SPEED", "EVENT_TYPE", "STATE",
         "DURATION_OF_STORM", "CZ_TIMEZONE","WINDY_EVENT","YEAR","MONTH_NAME",'CZ_NAME'])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("A plot:")

        # self.checkbox1 = QCheckBox('Show Regression Line', self)
        # self.checkbox1.stateChanged.connect(self.update)

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Index for subplots"))
        self.layout.addWidget(self.dropdown1)
        # self.layout.addWidget(self.checkbox1)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax1.clear()
        cat1 = self.dropdown1.currentText()

        numerical_features = ["WIND_SPEED", "DURATION_OF_STORM","WINDY_EVENT"]

        if cat1 in numerical_features:

            X_1 = ff_noaa["TOTAL_DAMAGE"]
            y_1 = ff_noaa[cat1]

            X_1 = np.log(X_1)

            # if cat1 in ['DAMAGE_CROPS']:
            #     y_1 = np.log(y_1)

            self.ax1.scatter(X_1,y_1)

            # if self.checkbox1.isChecked():
            #
            #     X_1.dropna(inplace=True)
            #     y_1.dropna(inplace=True)
            #
            #     b, m = polyfit(X_1, y_1, 1)
            #
            #     self.ax1.plot(X_1, b + m * X_1, '-', color="orange")

            vtitle = "TOTAL_DAMAGE vrs "+ cat1
            self.ax1.set_title(vtitle)
            self.ax1.set_xlabel("TOTAL_DAMAGE")
            self.ax1.set_ylabel(cat1)
            self.ax1.grid(True)

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

        elif cat1 in ['MONTH_NAME','YEAR']:

            df_1 = ff_noaa[[cat1, 'TOTAL_DAMAGE']]
            df_1 = df_1.groupby(cat1).sum().reset_index()
            df_1["TOTAL_DAMAGE"] = np.log(df_1["TOTAL_DAMAGE"])
            # df_1 = df_1.sort_values(by=['TOTAL_DAMAGE'], ascending=False)
            X_1 = df_1["TOTAL_DAMAGE"]
            y_1 = df_1[cat1]

            self.ax1.plot(y_1, X_1)

            vtitle = "TOTAL_DAMAGE vrs " + cat1
            self.ax1.set_title(vtitle)
            self.ax1.set_xlabel(cat1)
            self.ax1.set_ylabel("TOTAL_DAMAGE")
            self.ax1.grid(True)

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

        else:

            df_1 = ff_noaa[[cat1,'TOTAL_DAMAGE']]
            df_1 = df_1.groupby(cat1).sum().reset_index()
            df_1["TOTAL_DAMAGE"] = np.log(df_1["TOTAL_DAMAGE"])
            df_1 = df_1.sort_values(by=['TOTAL_DAMAGE'],ascending=False)
            X_1 = df_1["TOTAL_DAMAGE"][:10]
            y_1 = df_1[cat1][:10]

            self.ax1.bar(x =  y_1, height = X_1)

            vtitle = "TOTAL_DAMAGE vrs " + cat1
            self.ax1.set_title(vtitle)
            self.ax1.set_xlabel(cat1)
            self.ax1.set_ylabel("TOTAL_DAMAGE")
            self.ax1.grid(True)

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()


class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'NOAA'
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        EDA1Button = QAction(QIcon('analysis.png'),'Initial Assessment', self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('analysis.png'), 'Graphs wrt Total Damage', self)
        EDA2Button.setStatusTip('Graphs wrt Total Damage')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Correlation Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       XGBoost
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button =  QAction(QIcon(), 'XGBoost', self)
        MLModel1Button.setStatusTip('XGBoost ')
        MLModel1Button.triggered.connect(self.XGBoost)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)

        self.dialogs = list()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the happiness.score
        # X was populated in the method data_noaa()
        # at the start of the application
        #::------------------------------------------------------
        dialog = CanvasWindow(self)
        dialog.m.plot()
        x = (X - np.mean(X))/np.std(X)
        dialog.m.ax.hist(x,bins = 5, density=True, facecolor='green')
        # dialog.m.ax.distplot(x)
        # dialog.m.ax.X.plot(kind="hist", density=True, bins=15)
        # dialog.m.ax.X.plot(kind="kde")
        dialog.m.ax.set_title('Density of Total Damage')
        dialog.m.ax.set_xlabel("Total Damage")
        dialog.m.ax.set_ylabel("Density")
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This function creates an instance of DPGraphs class
        # This class creates a graph using the features in the dataset
        # happiness vrs the score of happiness
        #::---------------------------------------------------------
        dialog = DPGraphs()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def XGBoost(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = XGBoost()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def data_noaa():
    #::--------------------------------------------------
    # Loads the dataset 2017.csv ( Index of happiness and esplanatory variables original dataset)
    # Loads the dataset final_happiness_dataset (index of happiness
    # and explanatory variables which are already preprocessed)
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global noaa
    global ff_noaa
    global X
    global y
    global features_list
    global class_names
    global df_train
    global loaded_model_rf
    global loaded_model_xgb
    ff_noaa = pd.read_pickle('Data/cleaned_NAN_removed.pkl')
    X = ff_noaa["TOTAL_DAMAGE"]
    y = ff_noaa["STATE"]
    df_train = pd.read_pickle('Data/df_train.pkl')
    loaded_model_rf = pickle.load(open('Data/RF_Model.pkl','rb'))
    loaded_model_xgb = pickle.load(open('Data/XGB_Model.pkl', 'rb'))
    features_list = ["MAGNITUDE_TYPE", "WIND_SPEED", "EVENT_TYPE", "STATE",
         "DURATION_OF_STORM", "CZ_TIMEZONE","WINDY_EVENT","YEAR","MONTH_NAME",'CZ_NAME']

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    data_noaa()
    main()