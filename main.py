import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics

data=pd.read_csv("Advertising.csv", usecols=[ 'TV','radio','newspaper','sales'])
df=pd.DataFrame(data)
print(df)


df['TV'] = df['TV'].astype(int)
df['newspaper'] = df['newspaper'].astype(int)
df['radio'] = df['radio'].astype(int)
df['sales'] = df['sales'].astype(int)

print(df.dtypes)

sns.boxplot(data=df,x='TV')
plt.show()
sns.boxplot(data=df,x='newspaper')
plt.show()
sns.boxplot(data=df,x='radio')
plt.show()
sns.boxplot(data=df,x='sales')
plt.show()


# IQR


Q1 = df['sales'].quantile(0.25)

Q3 = df['sales'].quantile(0.75)

IQR = Q3 -  Q1

lower_bound1 = Q1 - 1.5 * IQR
upper_bound1= Q3 + 1.5 * IQR

df=df[(df['newspaper'] >= lower_bound1 ) & (df['newspaper']<= upper_bound1 )]

df=df[(df['sales'] >= lower_bound1) & (df['sales']<= upper_bound1)]

sns.boxplot(data=df,x='sales')
plt.show()

sns.boxplot(data=df,x='newspaper')
plt.show()

X=df.drop('sales',axis=1)
Y=df['sales']

sns.pairplot(df)
plt.show()

df['total'] = df['TV']+df['radio']+df['newspaper']


sns.pairplot(df)
plt.show()

ss=StandardScaler()
X_rescale=ss.fit_transform(X)


mms=MinMaxScaler()
X_mms_scale=mms.fit_transform(X)

X_train , X_test, Y_train , Y_test = train_test_split ( X,Y , test_size=0.10 random_state=0)

model=LinearRegression()
model.fit(X_train ,Y_train)
y_pred = model.predict(X_test)

print(y_pred)



mse  = mean_squared_error(Y_test ,y_pred)
mae = mean_absolute_error(Y_test ,y_pred)
rmse=np.sqrt(mse)
r2score= r2_score(Y_test,y_pred)


print(mse)
print(mse)
print(rmse)
print(r2score)


y_residual= Y_test - y_pred
print(y_residual)


sns.scatterplot(x=Y_test , y=y_residual)
plt.axhline(y= 0 , color = 'r' , linestyle ='--' )
plt.axhline(y= 2 , color = 'r' , linestyle ='--' )
plt.axhline(y= -2 , color = 'r' , linestyle ='--' )
plt.show()



final_model = LinearRegression()
final_model.fit(X.values,Y)
print(model.coef_)

new_data = [[35,25,41]]

print(final_model.predict(new_data))

test_residual = Y_test - y_residual
print(test_residual)
###############################################################   Diployment   ##############################################################
from joblib import dump,load

from tkinter import *
from tkinter import  messagebox as msg
from tkinter import ttk as ttk

my_form = Tk()
my_form.title('Linear Regression Model: ')
my_form.geometry('400x220')
my_form.resizable(False, False)
my_form.resizable(False, False)

def predict_sales(*args):
    tv = float(txt_TV.get())
    radio = float(txt_Radio.get())
    newspaper = float(txt_newspaper.get())

    new_data = [[tv, radio, newspaper]]
    loaded_model = load('final_model.pkl')
    msg.showinfo('Result: ',f'Result: {round(loaded_model.predict(new_data)[0],2)}')

lbl_TV = Label(my_form, text='TV')
lbl_TV.grid(row=0, column=0,padx=20,pady=10,sticky='w')

txt_TV = StringVar()
entry_TV = ttk.Entry(my_form, width=40, textvariable=txt_TV)
entry_TV.grid(row=0, column=1,padx=20,pady=10, sticky='w')

lbl_Radio = Label(my_form, text='Radio')
lbl_Radio.grid(row=1, column=0,padx=20,pady=10, sticky='w')

txt_Radio = StringVar()
entry_Radio = ttk.Entry(my_form, width=40,textvariable=txt_Radio)
entry_Radio.grid(row=1, column=1,padx=20,pady=10,  sticky='w')

lbl_newspaper = Label(my_form, text='newspaper')
lbl_newspaper.grid(row=2, column=0,padx=20,pady=10, sticky='w')

txt_newspaper = StringVar()
entry_newspaper = ttk.Entry(my_form, width=40, textvariable=txt_newspaper)
entry_newspaper.grid(row=2, column=1,padx=20,pady=10,  sticky='w')

btn_predict = ttk.Button(my_form, width=40, text='Predict Sales', command=predict_sales)
btn_predict.grid(row=3, column=1,padx=20,pady=10, sticky='w')

my_form.bind('<Return>', predict_sales)
my_form.mainloop()





