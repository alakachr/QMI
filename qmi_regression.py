# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 19:24:10 2021

@author: alaka
"""

import pandas as pd
import numpy as np, matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import pickle

x_train =pd.read_csv('C:/Users/alaka/Documents/QMI/ref_train_x.csv',  sep=';' , decimal=',') 

y_train = pd.read_csv('C:/Users/alaka/Documents/QMI/ref_train_y.csv',  sep=';' , thousands=',') 

# Lets see how the data looks like
print (x_train.head())




feat_imp = pd.DataFrame({'importance':rf.feature_importances_})    
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=True, inplace=True)

feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title= 'RandomForest Importance', figsize=(8,20))
plt.xlabel('Feature Importance Score')
plt.show()