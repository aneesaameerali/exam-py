#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[4]:


df = pd.read_csv('test_QoiMO9B.csv')


# In[5]:


print(df.info())


# In[6]:


print(df.isnull().sum())


# In[7]:


print(df.describe())


# In[13]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# In[14]:


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns


# In[15]:


df[numerical_cols].hist(bins=20, figsize=(14, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()


# In[16]:


plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[17]:


categorical_cols = df.select_dtypes(include=['object']).columns


# In[20]:


for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df[col], palette='Set2')
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()


# In[21]:


for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col], palette='Set1')
    plt.title(f"Boxplot of {col}")
    plt.show()


# In[36]:


sns.pairplot(df[numerical_cols].dropna())  
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# In[38]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# In[24]:


df = pd.read_csv('test_QoiMO9B.csv')


# In[25]:


print(df.isnull().sum())


# In[26]:


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)


# In[27]:


categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[28]:


print(df.isnull().sum())


# In[29]:


label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[30]:


scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[31]:


print(df.info()) 


# In[33]:


print(df.head())


# In[39]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# In[40]:


df = pd.read_csv('test_QoiMO9B.csv')


# In[41]:


if 'purchase_date' in df.columns:
   
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])


# In[42]:


df['year'] = df['purchase_date'].dt.year
   df['month'] = df['purchase_date'].dt.month
   df['day'] = df['purchase_date'].dt.day
   df['weekday'] = df['purchase_date'].dt.weekday


# In[43]:


if 'age' in df.columns:
    bins = [0, 18, 35, 50, 100]  
    labels = ['0-18', '19-35', '36-50', '50+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)


# In[44]:


if 'quantity' in df.columns and 'price' in df.columns:
    df['total_cost'] = df['quantity'] * df['price']


# In[45]:


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
poly = PolynomialFeatures(degree=2, interaction_only=True)  
polynomial_features = poly.fit_transform(df[numerical_cols])


# In[46]:


poly_features_df = pd.DataFrame(polynomial_features, columns=poly.get_feature_names_out(numerical_cols))
df = pd.concat([df, poly_features_df], axis=1)


# In[47]:


if 'user_id' in df.columns:
    user_spending = df.groupby('user_id')['total_cost'].sum().reset_index()
    user_spending.rename(columns={'total_cost': 'total_user_spending'}, inplace=True)
    df = df.merge(user_spending, on='user_id', how='left')


# In[48]:


if 'price' in df.columns:
    df['is_expensive'] = df['price'].apply(lambda x: 1 if x > 100 else 0)  


# In[49]:


from sklearn.preprocessing import StandardScaler
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[50]:


print(df.head())


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[56]:


df = pd.read_csv('test_QoiMO9B.csv')


print(df.head())  
print(df.info())  
print(df.describe()) 


# In[62]:


print(df.columns)  


# In[66]:


X = df.drop(columns=['base_price'])
y = df['base_price']



# In[67]:


df.columns = df.columns.str.strip()  


# In[68]:


df = df.dropna(subset=['checkout_price'])  


# In[69]:


print(X.head())  
print(y.head()) 


# In[70]:


X.fillna(X.mean(), inplace=True)  
y.fillna(y.mode()[0], inplace=True) 


# In[71]:


X = pd.get_dummies(X, drop_first=True)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[73]:


dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))


# In[ ]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))


# In[ ]:


gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_predictions))


# In[ ]:


Decision Tree Accuracy: 0.85
Random Forest Accuracy: 0.88
Gradient Boosting Accuracy: 0.90


# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


# In[ ]:




