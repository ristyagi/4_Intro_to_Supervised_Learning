import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = {
        'area' : [2600,3000,3200,3600,4000],
        'price' : [550000,565000,610000,680000,725000]
        }

data
type(data)

df = pd.DataFrame(data)
df
df.shape
df.info()
df.describe()

plt.scatter(df['area'], df['price'], color = 'Red', marker='+')
plt.xlabel('Area')
plt.ylabel('Price')

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price) # fitting linear regression model with 2D array of area 

reg.predict([[3300]]) # predicting price of new area by passing as 2D array


#internal details of lr

reg.coef_ # value of coeff calculated by model , 135.7876712
reg.intercept_ # value of intercept, 180616.43835616432


# y = mx + c
628715.75342466 = 135.7876712*3300 + 180616.43835616432

reg.predict([[5000]])

# house sizes to predict
sizes = [1000,1500,2300,3540,4120,4560,5490,3460,4750,2300,9000,8600,7100]
d = pd.DataFrame(sizes, columns=['area'])
d.head()
d.columns
type(d)

reg.predict(d[['area']])

# arr = d.to_numpy(d['area']) # converting single column df to numpy 2D array as that is required in reg.predict method
# arr

# d['predicted_price'] = reg.predict(arr)
d['predicted_price'] = reg.predict(d[['area']])
d.head()

d.to_csv("predicted_prices.csv", index=False)

reg.predict([[3000]])


######
plt.scatter(df['area'],df['price'],marker='+',color='Red')
plt.xlabel('Area(sqft)', fontsize = 20)
plt.ylabel('Price(USD)', fontsize = 15)
plt.plot(df['area'],reg.predict(df[['area']]),color = 'blue') # plotting liner line based on area and prices used to fit the model

