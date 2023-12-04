# %%
import pandas as pd
import numpy as np

data=pd.read_csv('weight-height.csv')
data.head(5)

# %%
# Detecting null values
print(data.isnull().sum()) #gives total null values per column

# %%
# Removing null values

df=data.dropna()  #deletes rows with null value
print(df.isnull().sum())

# %%
# Creating a DataFrame with null values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, np.nan, 10]
})
print(data)

# %%
# Imputing Null values with mean
imputed_data=data.fillna(data.mean())
print(imputed_data)


# %%
# Imputing Null values with specific value
imputed_data=data.fillna(0)
print(imputed_data)

# %%

# Forward fill
data_ffill = data.fillna(method='ffill')
print(data_ffill)

# %%

# Backward fill
data_bfill = data.fillna(method='bfill')
print(data_bfill)

# %%
# Interpolation

data_interpolated = data.interpolate(method='linear')
print(data_interpolated)

# %%
#numpy tensors

numpy_tensor=data_interpolated.to_numpy()
print(numpy_tensor)

# %%
#pytorch tensors
import torch

# Convert DataFrame to PyTorch tensor
torch_tensor = torch.tensor(data_interpolated.values)
print(torch_tensor)

# %%
#tensorflow tensors

import tensorflow as tf

# Convert DataFrame to TensorFlow tensor
tf_tensor = tf.constant(data_interpolated.values)
print(tf_tensor)