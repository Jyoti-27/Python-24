#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd


# In[3]:


import pandas as pd
#Create two series 
s1=pd.Series([6,7,8,9,5])
s2=pd.Series([0,1,2,3,4,5,7])
print('Series are : \n',s1, '\n', s2)


# In[4]:


print('Addition of series: \n', s1.add(s2))    #Elementwise addition
print('\n Subtraction of series: \n', s1.sub(s2))    #Elementwise Subtraction
print('\n Multiplication of series: \n', s1.mul(s2))
print('\n Division of series: \n', s1.div(s2))
print('Series are : \n',s1, '\n', s2)    #Series remains unchanged


# In[5]:


print(s1.append(s2))


# In[10]:


s3 = pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'])
print("Series is : \n", s3, '\n Indices are : ', s3.index)
print("Data type of   Series", type(s3.values)) 
s3


# In[11]:


s3.drop('c')


# In[11]:


print("\nMedian of series s2 is", s2.median())
print("\n Mean of series s2 is " , s2.mean())
print("\n Maximum of series s2 is", s2.max())
print("\n Minimum of series s2 is", s2.min())


# In[12]:


#Series with char/ string elements
string=pd.Series(['a','b','c','S','e','J','g','B','P','o'])
print('A Series wih String values: \n ', string)
print('string.str.upper(): \n',string.str.upper())
print('string.str.lower(): \n',string.str.lower())


# In[13]:


#Dataframe as a stack of Series.  we create two columns using series and then make a DataFrame
population_d= {'California': 3833, 'Texas': 8193,
                'New York': 6511, 'Florida': 5560, 'Ohio': 1135}    #Statewise population 
print(population_d, type(population_d))
population = pd.Series(population_d)
print(population)


# In[14]:


area_d = {'California': 423967, 'Texas': 695662, 'New York': 141297,   
             'Florida': 170312, 'Ohio': 149995}
area = pd.Series(area_d) 
print(area)


# In[15]:


states = pd.DataFrame({'Population': population,  'Area': area})
print("Data Frame of States: \n", states)


# In[18]:


states = pd.DataFrame({'Population': population_d,  'Area': area_d})
#print("Data Frame of States: \n", states)
states


# In[19]:


# Create a Dictionary from NumPy
import numpy as np
num_arr=np.random.randn(6,4)     #random delection of numbers following a standard normal distribution
print("Array is : \n", num_arr)  
cols=['A','B','C','D']            #arrays will not have index and columns
df1=pd.DataFrame(num_arr, columns=cols, index = ['i', 'ii', 'iii', 'iv', 'v', 'vi'])
#array of values, index, column
print('\n Data Frame from numpy array is : \n')
df1


# In[20]:


# create a dataframe using a dictionary of Lists, values are lists and column names are keys
data= {'city' : ['Bombay', 'Chennai', 'Chennai', 'Delhi', 'Mysore' ], 'year' : [2001, 2005, 2003, 2001, 2000],  
        'pop' : [25, 35, 20, 40, 15]}
df2= pd.DataFrame(data)
print(df2)
#observe index is assigned automatically


# In[25]:


import numpy as np
num_arr = np.random.randn(6,4)
print("Array is : \n", num_arr)


# In[37]:


cols=['A','B','C','D']            #arrays will not have index and columns
labels = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
#array of values, index, column
df = pd.DataFrame(num_arr, columns = cols, index = labels)
df


# In[36]:


cols=['A','B','C','D']            #arrays will not have index and columns
labels = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
#array of values, index, column
df = pd.DataFrame(num_arr, columns = cols, index = labels)
print(df)


# In[35]:


# create a dataframe using a dictionary of Lists, values are lists and column names are keys
data= {'city' : ['Bombay', 'Chennai', 'Chennai', 'Delhi', 'Mysore' ], 'year' : [2001, 2005, 2003, 2001, 2000],  
        'pop' : [25, 35, 20, 40, 15]}
print(data)
labels = ['A', 'B', 'C', 'D', 'E']
df2= pd.DataFrame(data, index=labels)
print(df2)
#observe index is assigned automatically


# In[38]:


# create a dataframe using a dictionary of Lists, values are lists and column names are keys
data= {'city' : ['Bombay', 'Chennai', 'Chennai', 'Delhi', 'Mysore' ], 'year' : [2001, 2005, 2003, 2001, 2000],  
        'pop' : [25, 35, 20, 40, 15]}
print(data)
labels = ['A', 'B', 'C', 'D', 'E']
df2= pd.DataFrame(data, index=labels)
df2
#observe index is assigned automatically


# In[39]:


print(df2.index)


# In[40]:


print(df2.columns)


# In[42]:


print(df2)


# In[43]:


df2


# In[44]:


print(df2.values)


# In[45]:


df2.values


# In[47]:


print(type(df2))


# In[50]:


print(df2)


# In[51]:


print(df2['city'])


# In[52]:


print(df2['year'])


# In[53]:


print(df2['A']) # To access a row of a dataframe loc(explicit) or iloc(implicit)


# In[54]:


print(df2.loc['A']) # To access a row of a dataframe loc(explicit) or iloc(implicit)


# In[55]:


print(df2.iloc['A']) # To access a row of a dataframe loc(explicit) or iloc(implicit)


# In[56]:


print(df2)
print("\n", df2.iloc[1])


# In[57]:


print(df2.loc['C', 'year'])


# In[58]:


print(df2.loc['C', 'year'])
print(df2.iloc[2,1])


# In[61]:


# Slicing
print(df2)
print([df2.iloc[1:3, 1:3]]) # Slicing with implicit index


# In[14]:


# create a dataframe using a dictionary of Lists, values are lists and column names are keys
data= {'city' : ['Bombay', 'Chennai', 'Chennai', 'Delhi', 'Mysore' ], 'year' : [2001, 2005, 2003, 2001, 2000],  
        'pop' : [25, 35, 20, 40, 15]}
print(data)
labels = ['A', 'B', 'C', 'D', 'E']
df2= pd.DataFrame(data, index=labels)
df2
#observe index is assigned automatically


# In[16]:


# Slicing
#print(df2)
print([df2.loc[1:3, 1:3]]) # Slicing with implicit index


# In[62]:


# Slicing
print(df2)
print([df2.iloc[1:3, 1:3]]) # Slicing with implicit index
df2.loc['B':'D', 'year':'pop']  #Slicing with explicit index


# In[63]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
animals_data=pd.DataFrame(data,index=labels)
animals_data
#observe the data type of each column


# In[64]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
#labels=['a','b','c','d','e','f','g','h','i','j']
animals_data = pd.DataFrame(data)
#animals_data=pd.DataFrame(data,index=labels)
animals_data
#observe the data type of each column


# In[69]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
#animals_data = pd.DataFrame(data)
animals_data=pd.DataFrame(data,index=labels)
animals_data
#observe the data type of each column


# In[66]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
#animals_data = pd.DataFrame(data)
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
#observe the data type of each column


# In[68]:


#DataFrame Attributes - index, cols, values, datatype of values¶
print("\n animals_data.index:\n ", animals_data.index)
print("\n animals_data.columns:\n", animals_data.columns)
print("\n animals_data.values:\n", animals_data.values)      #will show only values without index and column names
print("\n animals_data.dtypes:\n", animals_data.dtypes)    #will show the datatype of each column


# In[70]:


print(animals_data.head())   # will display top 5 lines of the dataFrame
print(animals_data.tail())     # will display bottom 5 lines of the dataframe


# In[71]:


print(animals_data.info())   # brief information about rows and columns


# In[72]:


print(animals_data.head(3))   # will display top 5 lines of the dataFrame
print(animals_data.tail(5))     # will display bottom 5 lines of the dataframe


# In[76]:


print("\n\n",animals_data.describe())   # Statistical discription of numeric columns 


# In[80]:


print(animals_data)
print(animals_data.describe())


# In[82]:


print(animals_data)
animals_data.describe()


# In[25]:


print(animals_data)
print(animals_data.describe())

#print(animals_data)
#print('\n'\nanimals_data.describe())


# In[83]:


# Information about the whole dataframe
print('\n Info : \n', animals_data.info())   #nrows, ncols, index, datatype of each column, number of nonnull values

#statistical data of dataframe
print('\n Statistical Description : \n',animals_data.describe())
#mean std max min quartiles for columns with numeric type
print('\n Description for object values: \n',animals_data.describe(include = ['object'])) 
#count, unique values, mode , freq


# In[84]:


print('\n Description for object values: \n', animals_data.describe(include = ['object']))
# count,unique,values,mode,frequency


# In[85]:


print(animals_data.T) # transpose


# In[88]:


# DataFrame Operations
# Sorting, Reindexing,Copy
# Original DataFrame is not modified
print(animals_data)
print("\n Sorting the data Agewise:\n", animals_data.sort_values(by = 'Age', ascending = False)) # sort by which column


# In[89]:


print("\n Sorting the data Agewise:\n", animals_data.sort_values(by = 'Age')) # sort by which column
print("\n Sorting the data Agewise:\n", animals_data.sort_index(axis = 1)) # sort operation for row


# In[90]:


print("\n Sorting the data Agewise:\n", animals_data.sort_values(by = 'Age')) # sort by which column
print("\n Sorting the data Agewise:\n", animals_data.sort_index(axis = 0)) # sort operation for column


# In[92]:


# Reindexing
print(animals_data)
label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
print(animals_data.reindex(label))
print(animals_data)


# In[93]:


# Reindexing
print(animals_data)
label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
animals_data.reindex(label)
animals_data


# In[94]:


# Reindexing
print(animals_data)
label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
print(animals_data.reindex(label).sort_index(axis=1))
print(animals_data)


# In[95]:


# Reindexing
print(animals_data)
label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
print(animals_data.reindex(label).sort_index(axis=0))
print(animals_data)


# In[133]:


# Reindexing
#print(animals_data)
#label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
label = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii','jj']
# Create a new DataFrame reindex takes a reaarangement of the existing index
print(animals_data.reindex(label).sort_index(axis=1))
print(animals_data)


# In[136]:


# Reindexing
print(animals_data)
#label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
label = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii','jj']
# Create a new DataFrame reindex takes a reaarangement of the existing ind
print(animals_data.reindex(label).sort_index(axis=1))
animals_data


# In[137]:


# Reindexing
print(animals_data)
#label = ['d', 'e', 'i', 'a', 'b', 'f', 'h', 'c', 'g','j']
label = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii','jj']
# Create a new DataFrame reindex takes a reaarangement of the existing ind
print(animals_data.reindex(label).sort_index(axis=0))
animals_data


# In[102]:


# Creating the copy of the data 
animals_data_c = animals_data.copy()
animals_data.copy()


# In[106]:


#Deleting a row or Column of a DataFrame
print(animals_data)
print("Drop rows with names 'a; and 'b':\n", animals_data.drop(['a'])) # dropping rows
print(animals_data)
print("Drop rows with names 'c; and 'b':\n", animals_data_c.drop(['c'], inplace = True)) # dropping rows
print(animals_data_c)


# In[124]:


#Deleting a row or Column of a DataFrame
#print(animals_data_c)
#print("Drop rows with names 'a; and 'b':\n", animals_data.drop(['a'])) # dropping rows
#print(animals_data)
print(animals_data)
print("Drop rows with names 'c': \n ", animals_data_c.drop(['c'], inplace = True)) # dropping rows
print(animals_data_c)


# In[119]:


print("Drop rows with names 'a', 'b' :\n", animals_data.drop(['a', 'b'],axis =0)) # dropping rows


# In[104]:


#Deleting a row or Column of a DataFrame
print(animals_data)
print("Drop rows with names 'a; and 'b':\n", animals_data.drop(['a', 'b']))
#dropping rows
#to drop the columns permanently use inplace - True
print("Drop rows with names 'a; and 'b':\n", animals_data_c.drop(['a', 'b'],
inplace=True))
print(animals_data.drop('Visits', axis=1))
#dropping column columns are axis=1 for drop() default is row
#So if we dont mention axis = 1 it will search for a row with name 'Visits'
print(animals_data.drop('Visits', axis='columns'))


# In[126]:


#Why doing an Aggregation on a Row doesnt make sense
print("Mean of the Dataframe is: \n",animals_data.mean()) #mean of values in columns containg numeric data
#columns containing numeric data
print("\nMean of 'Age' is: ",animals_data[['Age']].mean())
print("\nTotal visits :",animals_data[['Visits']].sum())
print("\nMax visits: ",animals_data[['Visits']].max())
print("\nMin visits: ",animals_data[['Visits']].min())
print("\n Index of Max visits: ",animals_data[['Visits']].idxmax()) # for what index is maximum
print("\n Index of Min visits: ",animals_data[['Visits']].idxmin())
print("\nSum: \n",animals_data.sum()) #for strings sum is string concatenation


# In[29]:


## Visualizing a DataFrames
#First Create a DataFrame
data={'Animals': ['cat','cat','turtle','dog','dog','cat','turtle','cat','dog','dog'],
          'Age': [2.5,3,0.5,np.nan,5,2,4.5,np.nan,7,3], 
      'Visits' : [1,3,2,3,2,3,1,1,2,1], 
    'Priority' : ['y','y','n','y','n','n','n','y','n','n']}
labels=['a','b','c','d','e','f','g','h','i','j']
#animals_data = pd.DataFrame(data)
animals_data=pd.DataFrame(data,index=labels)
print(animals_data)
#observe the data type of each column


# In[30]:


print("Mean of the Dataframe is: \n",animals_data.mean()) #mean of values in columns containg numeric data


# In[31]:


print("Mean Age of the Dataframe is: \n",animals_data[['Age']].mean()) #mean of values in columns containg numeric data


# In[ ]:




