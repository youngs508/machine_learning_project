
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random

class Utils:  
    def _updateTrainingData(self, x_train):
        for i in x_train.columns:
            if i != 'Date':
                x_train['yesterday_' +i] = x_train[i]
                x_train['twoDaysAgo_' +i] = x_train[i]
                x_train['threeDaysAgo_' +i] = x_train[i]
                x_train['fourDaysAgo_' +i] = x_train[i]
                x_train['fiveDaysAgo_' +i] = x_train[i]
                x_train['sixDaysAgo_' +i] = x_train[i]
                x_train['sevenDaysAgo_' +i] = x_train[i]
                x_train['yesterday_' +i] = x_train['yesterday_' +i].shift(1)
                x_train['twoDaysAgo_' +i] = x_train['twoDaysAgo_' +i].shift(2)
                x_train['threeDaysAgo_' +i] = x_train['threeDaysAgo_' +i].shift(3)
                x_train['fourDaysAgo_' +i] = x_train['fourDaysAgo_' +i].shift(4)
                x_train['fiveDaysAgo_' +i] = x_train['fiveDaysAgo_' +i].shift(5)
                x_train['sixDaysAgo_' +i] = x_train['sixDaysAgo_' +i].shift(6)
                x_train['sevenDaysAgo_' +i] = x_train['sevenDaysAgo_' +i].shift(7)
                x_train = x_train.drop(i, axis = 1)
        return x_train
    
    def _updateDate(self, x_train):
        x_train["Date"]
        x_train['Date'] = x_train['Date'].astype(str)
        
        for k in range (len(x_train["Date"])):
            size = len(x_train["Date"][k])
            string = x_train["Date"][k]
            substring_to_remove = ".0"
            x_train.loc[k]['Date'] = string.replace(substring_to_remove, "")

        for k in range (len(x_train["Date"])):
            size = len(x_train["Date"][k])
            string = x_train["Date"][k]
            x_train.loc[k]['Date'] = string[:4] + "-" + string[4:]
            #print(result)

        for k in range (len(x_train["Date"])):
            size = len(x_train["Date"][k])
            string = x_train["Date"][k]
            x_train.loc[k]['Date'] = string[:7] + "-" + string[7:]
        
        return x_train['Date']
    
    def _replaceNanY(self, y_train):
        column_means_7days =  y_train.tail(7).mean()
        df_filled = y_train_btcHigh.fillna(column_means_7days)
        y_train = df_filled
        
        return y_train
    
    def _replaceNanX(self, x_train):
        column_means = x_train.mean()
        df_filled = x_train.fillna(column_means)
        x_train = df_filled
        
        return x_train
    
    def _newTrainingData(self, new_row, x_train):
        index = len(x_train)
        x_train.loc[index] = new_row
        x_train_temp = x_train
        return x_train_temp

    
    def _newTrainLabel(self, y_train):
        column_means_7days = y_train.tail(7).mean()
        new_row = pd.Series({'btcHigh': column_means_7days})
        y_train_temp = y_train.append(new_row, ignore_index=True)
        return y_train_temp
    
    
    