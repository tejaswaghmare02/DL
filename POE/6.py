# %%
#Implement Forward pass with matrix multiplication 

# %%
import numpy as np

def activation(z):
  return 1/(1+np.exp(-z))

def loss_function(target,output):                #Mean Squared Error
  return (1/len(target))*np.square(target-output)

def forwardpass(x,weights,bias):
  weighted_sum=np.dot(x,weights)+bias
  print("Weighted Sum :\n",weighted_sum)
  output=activation(weighted_sum)
  return output

#input data
x=np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.9]])

#weights
weights=np.array([[0.8], [0.2]])

#targets
targets=np.array([[1], [0], [1]])

#bias
bias=np.array([0.1])

print("Input Data :\n",x)

output=forwardpass(x,weights,bias)
loss=loss_function(targets,output)

print("Output :\n",output)
print("Loss :\n",loss)


# %%
#Forward pass with hidden layer (matrix multiplication) 

# %%

def forwardpass(x,w1,w2,b1,b2):
  #input to hidden layer
  weighted_sum1=np.dot(x,w1)+b1
  print("Weighted Sum from input layer :\n",weighted_sum1)
  h_in=activation(weighted_sum1)
  #hidden to output layer
  weighted_sum2=np.dot(h_in,w2)+b2
  print("Weighted Sum from hidden layer :\n",weighted_sum2)
  h_out=activation(weighted_sum2)
  return h_out

#input data
x=np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.9]])

#targets
targets=np.array([[1], [0], [1]])

# Weights and biases for the input layer to hidden layer
w1= np.array([[0.8, 0.2], [0.4, 0.9]])
b1 = np.array([0.1, 0.5])

# Weights and bias for the hidden layer to output layer
w2 = np.array([[0.3], [0.7]])
b2= np.array([0.2])

final_output=forwardpass(x,w1,w2,b1,b2)
loss=loss_function(targets,final_output)

print("Input Data :\n",x)
print("Output :\n",final_output)
print("Loss :\n",loss)


# %%
#Forward pass with matrix multiplication with Keras 

# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def forward_pass(x, epochs=100):
    model = Sequential()
    model.add(Dense(units=1, activation='sigmoid', input_dim=2))  # Input to output
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, x, epochs=epochs)
    output = model.predict(x)
    return output

# Input data
x = np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.9]])

# Output data
output = forward_pass(x)

print("\nInput Data :\n", x)
print("\nOutput Data :\n", output)


# %%
#Forward passes with hidden layer (matrix multiplication) with Keras.

# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def forward_pass(x, epochs=100):
    model = Sequential()
    model.add(Dense(units=2, activation='sigmoid', input_dim=2))  # Input to hidden
    model.add(Dense(units=1, activation='sigmoid'))  # Hidden to output
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, x, epochs=epochs)
    output = model.predict(x)
    return output

# Input data
x = np.array([[0.5, 0.3], [0.2, 0.7], [0.8, 0.9]])

# Output data
output = forward_pass(x) 

print("\nInput Data :\n", x)
print("\nOutput Data :\n", output)
