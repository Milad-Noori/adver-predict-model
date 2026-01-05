import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from soupsieve.util import lower

data=pd.read_csv("Advertising.csv", usecols=[ 'TV','radio','newspaper','sales'])
df=pd.DataFrame(data)
# print(df)


df['TV'] = df['TV'].astype(int)
df['newspaper'] = df['newspaper'].astype(int)
df['radio'] = df['radio'].astype(int)
df['sales'] = df['sales'].astype(int)

# print(df.dtypes)

# sns.boxplot(data=df,x='TV')
# plt.show()
# sns.boxplot(data=df,x='newspaper')
# plt.show()
# sns.boxplot(data=df,x='radio')
# plt.show()


#IQR
Q1 = df['newspaper'].quantile(0.25)

Q3 = df['newspaper'].quantile(0.75)

IQR = Q3 -  Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df=df[(df['newspaper'] >= lower_bound ) & (df['newspaper']<= upper_bound )]

sns.boxplot(data=df,x='newspaper')
plt.show()

X=df.drop('sales',axis=1)
Y=df['sales']

ss=StandardScaler()
X_rescale=ss.fit_transform(X)


X_train , X_test, Y_train , Y_test = train_test_split(X_rescale,Y , test_size=0.15 , random_state=42)


model=LinearRegression()
model.fit(X_train ,Y_train)
y_pred = model.predict(X_test)

# print(y_pred)

from sklearn import metrics
mse  = mean_squared_error(Y_test ,y_pred)
mae = mean_absolute_error(Y_test ,y_pred)
rmse=np.sqrt(mse)
r2score= r2_score(Y_test,y_pred)


print(mse)
print(mse)
print(rmse)
print(r2score)