#!/usr/bin/env python
# coding: utf-8

# In[1]:


print ("Hello,Mark!"")


# In[2]:


print ("Hello,Mark!")


# In[3]:


Python

name= "Mark"
age= 30

print (name)
print (age)


# In[4]:


name="Mark"
age- 30 
print (name)
print (age)


# In[5]:


age= 30


# In[6]:


print(age)


# In[7]:


name = "Mark"
age = 30

print("My name is", name)
print("I am", age, "years old")


# In[8]:


height = 6.4
weight = 156

print(height)
print(weight)


# In[1]:


import pandas as pd


# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()


# In[3]:


iris.data


# In[4]:


iris.feature_names


# In[5]:


iris.target_names


# In[6]:


iris.target


# In[7]:


iris.data[5]


# In[11]:


iris.data[[0,1,2,3,4]]


# In[12]:


iris.target_names[iris.target[0]]


# In[13]:


iris.target_names[1]


# In[14]:


iris.target_names[2]


# In[15]:


iris.target[5]


# In[17]:


iris.target_names[iris.target[5]]


# In[18]:


df= pd.dataframe(iris.data, columns=iris.feature_names)


# In[19]:


import pandas as pd


# In[20]:


df=pd.dataframe(iris.data, columns=iris.feature_names)


# In[21]:


df=pd.DataFrame(iris.data, columns=iris.feature_names)


# In[22]:


df


# In[23]:


df["species"] = iris.target


# In[24]:


df


# In[25]:


x = df.drop("species", axis=1)


# In[26]:


y= df["species"]


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[39]:


from sklearn.tree import DecisionTreeClassifier


# In[40]:


from sklearn.tree import DecisionTreeClassifier


# In[41]:


model= DecisionTreeClassifier()


# In[44]:


model.fit(x_train, y_train)


# In[45]:


y_pred = model.predict(x_test)


# In[46]:


from sklearn.metrics import accuracy_score


# In[48]:


accuracy_score(y_test, y_pred)


# In[49]:


from sklearn.metrics import confusion_matrix 


# In[50]:


confusion_matrix(y_test, y_pred)


# In[51]:


model.predict([[5.1, 3.5, 1.4, 0.2]])


# In[54]:


species_name= ["Setosa","Versicolor", "Virginica"]
print("Predicted species", species_name[y_pred[0]])


# In[55]:


from sklearn.metrics import classification_report


# In[56]:


classification_report(y_test, y_pred)


# In[57]:


from sklearn.tree import plot_tree


# In[58]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[59]:


ConfusionMatrixDisplay.from_prediction(y_test, y_pred, display_labels=iris.target_names)


# In[60]:


import matplotlib.pyplot as plt


# In[62]:


plt.figure(figsize=(8, 6))


# In[63]:


plt.imshow(confusion_matrix(y_test, y_pred))


# In[64]:


plt.colorbar()


# In[ ]:




