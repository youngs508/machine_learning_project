from LinearRegressionRegularization import LinearRegressionRegularization
from Utils import Utils
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random

class Predictor: 
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.y_btcHigh_pred = []
        self.x_btcHigh_train = []
        self.y_btcHigh_train = []
        
    def getAllPredictionsBTCHigh(self):
        return self.y_btcHigh_pred
    
    def getAllXTrainBTCHigh(self):
        return self.x_btcHigh_train
    
    def getAllYTrainBTCHigh(self):
        return self.y_btcHigh_train
    
        
    def _predictWithLinearRegression(self, date=None):
    
        model = LinearRegressionRegularization()
        utils = Utils()
        
        #New training data
        if (date!=None): 
            self.X = utils._newTrainingData({'Date': date}, self.X)
            
        # Split the data into training subsets
        train_data, test_data, = model._split(self.X, self.y, ratio=0.7, random_state=123 )
        X_train_btcHigh = train_data.drop(columns=['btcHigh'])
        y_train_btcHigh = train_data['btcHigh']
        #Adjust training data 
        X_train_btcHigh = utils._updateTrainingData(X_train_btcHigh)
        X_train_btcHigh = utils._replaceNanX(X_train_btcHigh)
        has_nan = X_train_btcHigh.isna().any().any()
        if (has_nan != True):
            #Predict
            y_pred_btcHigh = model._predict(X_train_btcHigh, y_train_btcHigh)
        else:
            print ("Training Data contains Nan values, ", has_nan)
        pd.options.mode.chained_assignment = None  #Hide warning
        X_train_btcHigh["Date"] = utils._updateDate(X_train_btcHigh)
        
        self.y_btcHigh_pred = y_pred_btcHigh 
        self.x_btcHigh_train = X_train_btcHigh 
        self.y_btcHigh_train = y_train_btcHigh 
        
        return y_pred_btcHigh.tail(1).to_string(index=False)
