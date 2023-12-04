# %%
# 1. Implement Data Manipulation (Numpy library) Operations Broadcasting Indexing and slicing. Also display head, tail, size of data and remove null values from the dataset.

# %%
# Broadcasting
import numpy as np

#create a numpy array
arr1=np.arange(1,10)
print(arr1)

b=5
result=arr1*b
print(result)

# %%
# Indexing

arr2 = np.array([10, 20, 30, 40, 50])

element = arr2[2]  # Accessing the third element
print(element)

# %%
# Slicing 
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

row_slice = matrix[1, :]  # Extracting the second row
print(row_slice)

column_slice = matrix[:, 1]  # Extracting the second column
print(column_slice)

submatrix = matrix[0:2, 1:]  # Extracting a submatrix from rows 1 and 2, and columns 2 and 3
print(submatrix)

# %%
# read csv
import pandas as pd

data=pd.read_csv('weight-height.csv')

# %%
#head
print(data.head())


# %%
#tail
print(data.tail())

# %%
#size of data
print(data.size)


# %%
# Detecting null values

print(data.isnull().sum()) #gives total null values per column


# %%
# Removing null values

df=data.dropna()  #deletes rows with null value
print(df.isnull().sum())

# %%
print(data.notnull().sum()) # not null values