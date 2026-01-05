import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import lineStyles
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# sns.boxplot(data=df,x='sales')
# plt.show()


#IQR
Q1 = df['newspaper'].quantile(0.25)

Q3 = df['newspaper'].quantile(0.75)

IQR = Q3 -  Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

Q1 = df['sales'].quantile(0.25)

Q3 = df['sales'].quantile(0.75)

IQR = Q3 -  Q1

lower_bound1 = Q1 - 1.5 * IQR
upper_bound1= Q3 + 1.5 * IQR

df=df[(df['newspaper'] >= lower_bound ) & (df['newspaper']<= upper_bound )]

df=df[(df['sales'] >= lower_bound1) & (df['sales']<= upper_bound1)]

# sns.boxplot(data=df,x='sales')
# plt.show()

# sns.boxplot(data=df,x='newspaper')
# plt.show()

X=df.drop('sales',axis=1)
Y=df['sales']
#
# sns.pairplot(df)
# plt.show()

# df['total'] = df['TV']+df['radio']+df['newspaper']


# sns.pairplot(df)
# plt.show()

ss=StandardScaler()
X_rescale=ss.fit_transform(X)


# mms=MinMaxScaler()
# X_mms_scale=mms.fit_transform(X)

X_train , X_test, Y_train , Y_test = train_test_split(X,Y , test_size=0.15 , random_state=42)


model=LinearRegression()
model.fit(X_train ,Y_train)
y_pred = model.predict(X_test)

# print(y_pred)

from sklearn import metrics
mse  = mean_squared_error(Y_test ,y_pred)
mae = mean_absolute_error(Y_test ,y_pred)
rmse=np.sqrt(mse)
r2score= r2_score(Y_test,y_pred)


# print(mse)
# print(mse)
# print(rmse)
# print(r2score)
#
#
y_residual= Y_test - y_pred
# print(y_residual)


# sns.scatterplot(x=Y_test , y=y_residual)
# plt.axhline(y= 0 , color = 'r' , linestyle ='--' )
# plt.axhline(y= 2 , color = 'r' , linestyle ='--' )
# plt.axhline(y= -2 , color = 'r' , linestyle ='--' )
# plt.show()



final_model = LinearRegression()
final_model.fit(X.values,Y)
# print(model.coef_)

new_data = [[35,25,41]]

print(final_model.predict(new_data))


###############################################################   Diployment   ##############################################################
from joblib import dump,load


dump(final_model,'final_model.pkl')
load_model = load('final_model.pkl')



