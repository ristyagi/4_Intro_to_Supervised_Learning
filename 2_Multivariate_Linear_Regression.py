#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:54:56 2024

@author: Rishabh_Tyagi
"""

# https://www.youtube.com/watch?v=J_LnPL3Qg70&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math


base_data = {
    'area' : [2600,3000,3200,3600,4000],
    'bedrooms' : [3,4,np.nan,3,5],
    'age' : [20,15,18,30,8],
    'price' : [550000,565000,610000,595000,760000]
    }

base_data

df = pd.DataFrame(base_data)
df
df.head()
df.info()
df.describe()


# filling na values with floored median values
df['bedrooms'].median()

math.floor(df['bedrooms'].median())

df['bedrooms'] = df['bedrooms'].fillna(math.floor(df['bedrooms'].median()))
df


mlr = linear_model.LinearRegression()

mlr.fit(df[['area','bedrooms','age']],df['price'])

mlr.coef_

mlr.intercept_

mlr.predict([[3000,3,40]])
mlr.predict([[2500,4,5]])
mlr.predict([[3000,4,15]])


# how calc is done for mlr.predict([[3000,3,40]])
# 137.25*3000 + -26025*3 + -6825*40 + 383725
