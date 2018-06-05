import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')





data = pd.read_csv("FlightDelays.csv")




data.head()



data.describe()




print(data.dtypes)
print(data.shape)


# ## Cleaning the Data




## Converting all Categorical Data into Category type 
data.carrier = data.carrier.astype("category")
data.origin = data.origin.astype("category")
data.weather = data.weather.astype("category")
data.dayweek = data.dayweek.astype("category")
data.daymonth = data.daymonth.astype("category")



## Dropping Variables which are considered Not Significant
X = data.drop(["tailnu","date","daymonth"],axis=1)





X.dtypes




## creating Dummy variables 
X_dum = pd.get_dummies(X)





X_dum.head()



X_delayed = X_dum.drop(["delay_ontime"], axis=1)




X_delayed.head()



## Splitting Independent and dependent variables 
x = X_delayed.drop(["delay_delayed"],axis=1)
y = X_delayed["delay_delayed"]


# # Building Classification Models 

# #### All the models are checked for their acuuracy using cross validation score where you divide the whole data set into no of cross validations where no of cv is equal to number of equal split it is done 

# #### Using Decision Tree Classifier



from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


from sklearn.cross_validation import cross_val_score


### Decision Tree 
dtree = DecisionTreeClassifier(max_depth=5, min_samples_split=2 )


dtree.fit(x,y)

cross_val_score(dtree,x,y,cv=8)


# #### Using Random Forest Classifier 


## RandomForest
rforest = RandomForestClassifier(n_estimators=100, max_depth=10,min_samples_split=2)


rforest.fit(x,y)


cross_val_score(rforest,x,y,cv=8)


# #### Using Gradient Boosting Classifier 

# In[25]:


## Gradient Boosting Classifier
GBCl = GradientBoostingClassifier(learning_rate=0.01,n_estimators=100, max_depth=10,min_samples_split=2)


# In[26]:


cross_val_score(GBCl,x,y,cv=8, n_jobs=-1)

