import pandas as pd
from sklearn import linear_model

df=pd.read_csv('Multiple_Linear_Regression.csv')

Reg=linear_model.LinearRegression()

Reg.fit(df[['area','bedroom','old']],df.price)

# Price prediction of a 2000 square meter house with 3 rooms and 10 years old

print(Reg.predict([[2000,3,10]]))

# Price prediction of a 1500 square meter house with 4 rooms and 5 years old

print(Reg.predict([[1500 , 4 , 5]]))