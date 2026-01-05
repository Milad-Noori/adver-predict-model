import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data=pd.read_csv("Advertising.csv", usecols=[ 'TV','radio','newspaper','sales'])
df=pd.DataFrame(data)
print(df)



X=df.drop('sales',axis=1)
Y=df['sales']

ss=StandardScaler()
X_rescale=ss.fit_transform(X)


X_train , X_test, Y_train , Y_test = train_test_split(X_rescale,Y , test_size=0.2 , random_state=42)


model=LinearRegression()
model.fit(X_train ,Y_train)
y_pred = model.predict(X_test)

print(y_pred)