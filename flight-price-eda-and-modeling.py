#!/usr/bin/env python
# coding: utf-8

# ## Loading Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso, RidgeCV,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


# ## Loading the training and the testing dataset

# In[2]:


train = pd.read_excel('Data_Train.xlsx')
train.head()


# In[3]:


test = pd.read_excel('Test_Set.xlsx')
test.head()


# In[4]:


print("Training Dataset Shape:",train.shape)
print("Test Dataset Shape:",test.shape)


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train.describe()


# In[8]:


train.isna().sum()


# In[9]:


test.isna().sum()


# #### Observation:
# 
# There are two null values, let's observe them

# In[10]:


train[train.isnull().any(axis=1)]


# #### Observation:
# 
# Both of them are from the same row which is quite less to impute, we can drop the null row

# In[11]:


train.dropna(inplace = True)


# ## Exploratory Data Analysis

# In[12]:


print('Observing the airline column:')
train['Airline'].value_counts()


# In[13]:


plt.figure(figsize=(8,3))
sns.countplot(x='Airline',data=train)
plt.title('Airline Flight Count')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# #### Observation:
# 
# There is a lot more data on Jetairways which could lead to a slight bias in the model. There occurances for Vistara Premium economy, Jet Airways Business, Multiple Carriers Premium Economy, Truejet is quite less - this could add some noise to the model. We shall deal with it later

# In[14]:


print('Observing the Price column for each airline: ')
unique_airlines = train['Airline'].unique()
n = 6
num_rows = int(np.ceil(len(unique_airlines) / n))

plt.figure(figsize=(20, 5 * num_rows))

for i, airline in enumerate(unique_airlines):
    plt.subplot(num_rows,n,i + 1)
    sns.boxplot(data=train[train['Airline']==airline], x='Airline', y='Price')
    plt.title(airline)
    plt.xlabel('')
    plt.ylabel('Price' if i % n == 0 else '')
plt.show()


# #### Observation:
# 
# There are quite a few price outliers for Indigo, AirIndia, Jet Airways, SpiceJet. It could be be because of the number of flights for each of the flight and the different routes and destinations. We shall explore this later

# In[15]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Total_Stops',y='Price',data=train)
plt.title('Stops Vs Ticket Price')
plt.xlabel('Stops')
plt.ylabel('Price')
plt.show()


# In[16]:


plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
sns.countplot(x='Destination',data=train)
plt.title('Airline Flight Count')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.boxplot(x='Destination',y='Price',data=train)
plt.title('Destination Vs Ticket Price')
plt.xlabel('Destination')
plt.ylabel('Price')

plt.show()


# #### Observation:
# 
# We can see that the outliers are quite high for Cochin because of the number of flights headed across multiple locations. For New Delhi and Hyderabad, it could be because of their metropolitan status.
# 
# We can also observe that New Delhi and Delhi are being treated differently here, this will be dealt with later

# ## Feature Engineering

# In[17]:


def classify_flight_time(time_str):
    time = datetime.strptime(time_str, '%H:%M').time()
    if time >= datetime.strptime('00:00', '%H:%M').time() and time < datetime.strptime('06:00', '%H:%M').time():
        return 'Early Morning'
    elif time >= datetime.strptime('06:00', '%H:%M').time() and time < datetime.strptime('12:00', '%H:%M').time():
        return 'Morning'
    elif time >= datetime.strptime('12:00', '%H:%M').time() and time < datetime.strptime('18:00', '%H:%M').time():
        return 'Afternoon'
    elif time >= datetime.strptime('18:00', '%H:%M').time() and time < datetime.strptime('22:00', '%H:%M').time():
        return 'Evening'
    else:
        return 'Late Night'

train['Departure'] = train['Dep_Time'].apply(classify_flight_time)
test['Departure'] = test['Dep_Time'].apply(classify_flight_time)


# In[18]:


def calculate_arrival(departure, duration):
    departure_time = datetime.strptime(departure, "%H:%M")    
    if 'h' in duration:
        parts = duration.split('h')
        hours = int(parts[0]) if parts[0] else 0
        minutes = int(parts[1].strip().replace('m', '')) if parts[1] else 0
    else:
        hours = 0
        minutes = int(duration.replace('m', ''))
    arrival_time = departure_time + timedelta(hours=hours, minutes=minutes)
    return arrival_time.strftime("%H:%M")

train['Arr_Time'] = train.apply(lambda row: calculate_arrival(row['Dep_Time'], row['Duration']), axis=1)
train['Arrival'] = train['Arr_Time'].apply(classify_flight_time)

test['Arr_Time'] = test.apply(lambda row: calculate_arrival(row['Dep_Time'], row['Duration']), axis=1)
test['Arrival'] = test['Arr_Time'].apply(classify_flight_time)


# In[19]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.countplot(x='Departure',data=train)
plt.title('Departure Time Flight Count')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.boxplot(x='Departure',y='Price',data= train)
plt.title('Departure Time vs Price')
plt.xlabel('Departure Time')
plt.ylabel('Price')
plt.show()


# In[20]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.countplot(x='Arrival',data=train)
plt.title('Arrival Time Flight Count')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.boxplot(x='Arrival',y='Price',data= train)
plt.title('Arrival Time vs Price')
plt.xlabel('Arrival Time')
plt.ylabel('Price')
plt.show()


# #### Observation:
# 
# Departure Time: There are quite a few outliers for Morning and Afternoon departure, which makes sense as the number of flights departing at the time are quite high. Even though there are a few outliers for Morning, they aren't that much.
#     
# Similarly for Arrival Time, there are quite a few outliers for Evening, Morning, and Afternoon departure as the number of flights arriving at the time are quite high. 

# In[21]:


train['Date'] = pd.to_datetime(train['Date_of_Journey'], format='%d/%m/%Y')
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['Year'] = train['Date'].dt.year

test['Date'] = pd.to_datetime(test['Date_of_Journey'], format='%d/%m/%Y')
test['Month'] = test['Date'].dt.month
test['Day'] = test['Date'].dt.day
test['Year'] = test['Date'].dt.year


# In[22]:


train['Year'].value_counts()


# In[23]:


plt.figure(figsize=(6,3))
sns.countplot(x='Month',data=train)
plt.title('Month Flight Count')
plt.ylabel('Count')
plt.show()


# #### Observation:
# 
# There are a lot more flights being taken in the summer (May and June)

# In[24]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Month',y='Price',data=train)
plt.title('Month vs Ticket Price')
plt.xlabel('Month')
plt.ylabel('Price')
plt.show()


# In[25]:


def extract_time_components(df, time_column1, time_column2, format='%H:%M'):
    df[time_column1] = pd.to_datetime(df[time_column1], format=format)
    df[time_column2] = pd.to_datetime(df[time_column2], format=format)

    df['Dept_Hour'] = df[time_column1].dt.hour
    df['Dept_Minute'] = df[time_column1].dt.minute
    
    df['Arr_Hour'] = df[time_column2].dt.hour
    df['Arr_Min'] = df[time_column2].dt.minute

    return df

train = extract_time_components(train, 'Dep_Time', 'Arr_Time')
test = extract_time_components(test, 'Dep_Time', 'Arr_Time')


# In[26]:


train['Airline'].value_counts()


# In[27]:


test['Airline'].value_counts()


# #### Observation:
# 
# To deal with the Airline categories having lesser counts, we'll group them into an "Other" category to reduce noise

# In[28]:


replace_with_others = ['Multiple carriers Premium economy', 'Jet Airways Business','Vistara Premium economy', 'Trujet']

train['Airline'] = train['Airline'].replace(replace_with_others, 'Others')
test['Airline'] = test['Airline'].replace(replace_with_others, 'Others')


# In[29]:


train['Additional_Info'].value_counts()


# In[30]:


test['Additional_Info'].value_counts()


# #### Observation:
# 
# Similar to what we did for Airline, we'll group smaller Additional Info cats to reduce noise

# In[31]:


train['Additional_Info'].replace({'Change airports':'Other', 
                                  'Business class':'Other',
                                  '1 Short layover':'Other',
                                  '1 Long layover':'Other',
                                  'Red-eye flight':'Other',
                                  '2 Long layover':'Other',
                                  'No info':'No Info'},inplace=True)

test['Additional_Info'].replace({'Change airports':'Other', 
                                 'Business class':'Other',
                                 '1 Short layover':'Other',
                                 '1 Long layover':'Other',
                                 'Red-eye flight':'Other',
                                 '2 Long layover':'Other',
                                 'No info':'No Info'},inplace=True)


# In[32]:


train['Additional_Info'].value_counts()


# In[33]:


test['Additional_Info'].value_counts()


# #### Observation:
# 
# Mapping New Delhi to Delhi in both Source and Destination to avoid inconsistency

# In[34]:


train['Source'].replace({'New Delhi':'Delhi'},inplace=True)
test['Source'].replace({'New Delhi':'Delhi'},inplace=True)

train['Destination'].replace({'New Delhi':'Delhi'},inplace=True)
test['Destination'].replace({'New Delhi':'Delhi'},inplace=True)


# In[35]:


def duration_to_minutes(duration):
    hours = 0
    minutes = 0
    if 'h' in duration:
        parts = duration.split('h')
        hours = int(parts[0]) if parts[0] else 0
        if len(parts) > 1 and parts[1]:
            minutes = int(parts[1].replace('m', ''))
    else:
        minutes = int(duration.replace('m', ''))
    total_minutes = hours * 60 + minutes
    return total_minutes

train['Duration_Min'] = train['Duration'].apply(duration_to_minutes)
test['Duration_Min'] = test['Duration'].apply(duration_to_minutes)


# In[36]:


plt.figure(figsize=(10,4))
sns.boxplot(x= train['Price'])
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.show()


# In[37]:


skewness = train['Price'].skew()
print(f"The skewness of the price distribution is: {skewness}")


# #### Observation:
# 
# The Skewness is quite high, Log transformation could help

# In[38]:


plt.figure(figsize=(10, 4))
sns.boxplot(x=train['Price'].apply(lambda x: np.log(x+1)))
plt.title('Boxplot of Log-transformed Price')
plt.xlabel('Log(Price)')
plt.show()


# #### Observation:
# 
# Log transformation would help in reducing outliers but there would still be a few. Let's winsorize!

# In[39]:


train['Log_Price'] = np.log1p(train['Price'])
train['Winsorized_Log_Price'] = winsorize(train['Log_Price'], limits=[0.05, 0.05])


# In[40]:


plt.figure(figsize=(5, 4))
sns.boxplot(x=train['Winsorized_Log_Price'])
plt.title('Boxplot of Winsorized_Log_Price')
plt.xlabel('Log(Price)')
plt.show()


# #### Observation:
# 
# Much better!

# In[41]:


train_cols_drop = ['Date_of_Journey','Dep_Time','Arrival_Time','Date','Arr_Time','Duration','Price','Log_Price','Year']
test_cols_drop = ['Date_of_Journey','Dep_Time','Arrival_Time','Date','Arr_Time','Duration','Year']
train = train.drop(train_cols_drop,axis=1)
test = test.drop(test_cols_drop,axis=1)


# In[42]:


plt.figure(figsize = (8,8))
sns.heatmap(train.corr(), annot = True)
plt.show()


# In[43]:


print("Observing the categorical column disribution before encoding: \n")

cols_cat = ['Airline','Source','Destination','Total_Stops','Additional_Info','Departure','Arrival']
for columns in cols_cat:
    print(columns, '\n')
    print(train[columns].value_counts(),'\n')


# In[44]:


train['Route'].value_counts()


# #### Observations:
# 
# Route has too many values, let's apply Frequency encoding

# In[45]:


route_counts = train['Route'].value_counts().to_dict()

train['Route_Frequency'] = train['Route'].map(route_counts)
test['Route_Frequency'] = test['Route'].map(route_counts)

train = train.drop('Route',axis=1)
test = test.drop('Route',axis=1)


# In[46]:


encoder = LabelEncoder()
for columns in cols_cat:
    train[columns] = encoder.fit_transform(train[columns])
    test[columns] = encoder.fit_transform(test[columns])

print("Observing the categorical column disribution after encoding: \n")    
for columns in cols_cat:
    print(columns, '\n')
    print(train[columns].value_counts(),'\n')


# In[47]:


correlation_matrix = train.corr()
target_correlations = correlation_matrix['Winsorized_Log_Price']
print(target_correlations)


# ## Modeling

# In[48]:


X = train.drop('Winsorized_Log_Price', axis=1)
y = train['Winsorized_Log_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[49]:


classifiers = [
    LinearRegression(),
    ElasticNet(),
    RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]),
    Lasso(alpha =16, random_state=100),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    SVR(), 
    xgb.XGBRegressor()]

name = []
score = []
models = []
rmse = []
r_2_score = []
i = 0
for classifier in classifiers:
    classifier.fit(X_train, y_train)   
    name.append(type(classifier).__name__)
    score.append(classifier.score(X_test, y_test))
    models.append(classifier)
    rmse.append(np.sqrt(mean_squared_error(classifier.predict(X_test), y_test)))
    r_2_score.append(r2_score(classifier.predict(X_test), y_test))


# In[50]:


df_score = pd.DataFrame(list(zip(name,rmse,r_2_score,score, models)),columns=['name','rmse','r2_score','score',"model"])
df_score.set_index('name',inplace=True)
df_score.sort_values(by=['score'],inplace=True)
df_score


# In[51]:


model = df_score.loc["XGBRegressor","model"]
predict = model.predict(test)
predict


# In[52]:


test['Prediction_Price'] = predict
test


# #### Observation:
# 
# Let's scaleback the price column

# In[53]:


test['Prediction_Price'] = np.exp(test['Prediction_Price'])
test

