
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random

class LinearRegressionRegularization:
    def __init__(self, max_iter=5,  learningRate=50, random_state=None):
        self.max_iter_ = max_iter
        self.alpha = learningRate
        self.random_state_ = random_state
        self.w_ = np.random.randint(0, 1, 17)
        self.w0 = 0
        
    def _split(self, X, y, ratio, random_state):
        header = X.columns
        # remove header for shuffling 
        x_data = X.values

        random.seed(random_state)
        train_ratio = ratio
        test_ratio = 1 - train_ratio
        
        total_data_sample = len(X)
        train_samples = int(total_data_sample * train_ratio)
        test_samples = total_data_sample - train_samples

        random.shuffle(x_data)
        shuffled_data = pd.DataFrame(x_data, columns=header)

        train_data = shuffled_data.head(train_samples)
        test_data = shuffled_data.head(test_samples)
        
        return train_data, test_data
        
    def _predict(self, x_train, y_train):
        #TO-DO : Add Regularization 
        
        x_train_array = x_train.to_numpy()
        m = len(y_train) # number of features 
        #Initial weights 
        tempWeights = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
        tempW0 = 0
        y_pred_btc = tempW0
        
        for idx in range (self.max_iter_): 
            for i in range (m):
                summation = []
                xi = x_train_array[i, :]
                y_pred_btc = tempW0
                for k in range (17):
                    y_pred_btc += self.w_[k] * xi[k]
                summation.append(y_pred_btc - y_train)

                derivative = 2/m * sum(summation)
                tempW0 = self.w0 - (self.alpha * derivative)

            for j in range (17):
                for i in range (m): 
                    summation = []
                    xi = x_train_array[i, :]
                    y_pred_btc = tempW0 
                    for k in range (17):
                        y_pred_btc += self.w_[k] * xi[k]  

                    summation.append(y_pred_btc - y_train)
                derivative = 2/m * sum(summation)
                tempWeights.append(self.w_[j] - (self.alpha * derivative))    

            #The previous iteration 
            last_17 = tempWeights[-17:]

            #Assigning new weight values to old weights
            for n in range (17):
                tempWeights[n] = last_17[n]
            self.w0 = tempW0
            #next iteration

        final_weights = tempWeights[:17]
        self.w_ = final_weights
        
        return y_pred_btc
    
