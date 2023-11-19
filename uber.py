import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("uber.csv")
#print(df.head())
#print(df.info)
#print(df.shape)  #20000,9
df.drop(['Unnamed: 0','key'],axis=1,inplace=True)
#print(df.shape)    #20000,7

#print(df.isnull().sum())
df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(),inplace=True)
df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(),inplace=True)
#print(df.isnull().sum())

#df.drop(df[df['fare_amount'].values<=0].index,inplace=True)
#print(df.shape)  #19978,7


df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
#print(df.dtypes)
df = df.assign(hour = df.pickup_datetime.dt.hour,
               day = df.pickup_datetime.dt.day,
               month = df.pickup_datetime.dt.month,
               year = df.pickup_datetime.dt.year,
               dayofweek = df.pickup_datetime.dt.dayofweek)

#print(df.head()) now 12 columns
df=df.drop('pickup_datetime',axis=1)

#Using the InterQuartile Range to fill the values
def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1

df = treat_outliers_all(df , df.iloc[: , 0::])

#print(df.shape) #199978,11
#df=treat_outliers(df,df.iloc[:,0::])
#print(df.shape) #148066,11

corr=df.corr()
#sns.heatmap(data=corr,annot=True)
#plt.show()

# Dividing the dataset into feature and target values 
x = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month','year','dayofweek']]
y = df['fare_amount']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)

#print(r2_score(y_test,pred))
#frame = pd.DataFrame({"Actual" : y_test , "Predicted" : pred})
#print(frame)
print("Mean Squared Error:",mean_squared_error(y_test,pred))
print("R^2 Score:",r2_score(y_test, pred))


rf_model = RandomForestRegressor()
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)

print("Mean Squared Error:",mean_squared_error(y_test,rf_predictions))
print("R^2 Score:",r2_score(y_test, rf_predictions))
