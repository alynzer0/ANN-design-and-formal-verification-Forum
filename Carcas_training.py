import numpy as np
#np.random.seed(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))
#def relu(x):
  #  return np.maximum(0.0, x)

#def drelu(x):
 #   return 1. * (x > 0.01)
def sigmoid_derivative(x):
    return x * (1 - x)


#def rectified_derivative (x):
 #   return x * (1 - x)
#Input datasets
inputs = np.array([[0.2,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
expected_output = np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

epochs = 100000
lr = 0.1
inputLayerNeurons, h1,h2,h3,outputLayerNeurons = 4,6,6,6,5

#Random weights and bias initialization
hidden_weights1 = np.random.uniform(size=(inputLayerNeurons,h1))
hidden_bias1 =np.random.uniform(size=(1,h1))

#hidden_weights2 = np.random.uniform(size=(h1,h2))
#hidden_bias2 =np.random.uniform(size=(1,h2))

#hidden_weights3 = np.random.uniform(size=(h1,h3))
#hidden_bias3 =np.random.uniform(size=(1,h3))

hidden_weights2 = np.random.uniform(size=(h1,h2))
hidden_bias2 =np.random.uniform(size=(1,h2))

hidden_weights3 = np.random.uniform(size=(h3,h2))
hidden_bias3 =np.random.uniform(size=(1,h3))

output_weights = np.random.uniform(size=(h3,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))

print("Initial hidden weights: ",end='')
print(*hidden_weights1)
print("Initial hidden biases: ",end='')
print(*hidden_bias1)
print("Initial hidden weights2: ",end='')
print(*hidden_weights2)
print("Initial hidden biases2: ",end='')
print(*hidden_bias2)
print("Initial output weights: ",end='')
print(*output_weights)
print("Initial output biases: ",end='')
print(*output_bias)


#Training algorithm
for _ in range(epochs):
	#Forward Propagation
 hidden_layer_activation1 = np.dot(inputs,hidden_weights1)
 hidden_layer_activation1 += hidden_bias1
 hidden_layer_output1 = sigmoid(hidden_layer_activation1)

 #hidden_layer_activation2 = np.dot(hidden_layer_output1,hidden_weights2)
 #hidden_layer_activation2 += hidden_bias2
 #hidden_layer_output2 = sigmoid(hidden_layer_activation2)

 #hidden_layer_activation3 = np.dot(hidden_layer_output1,hidden_weights3)
 #hidden_layer_activation3 += hidden_bias3
 #hidden_layer_output3 = sigmoid(hidden_layer_activation3)

 hidden_layer_activation2 = np.dot(hidden_layer_output1,hidden_weights2)
 hidden_layer_activation2 += hidden_bias2
 hidden_layer_output2 = sigmoid(hidden_layer_activation2)

 hidden_layer_activation3 = np.dot(hidden_layer_output2,hidden_weights3)
 hidden_layer_activation3 += hidden_bias3
 hidden_layer_output3 = sigmoid(hidden_layer_activation3)

 output_layer_activation = np.dot(hidden_layer_output3,output_weights)
 output_layer_activation += output_bias
 predicted_output = sigmoid(output_layer_activation)

  #Backpropagation
 error = expected_output - predicted_output
 d_predicted_output = error * sigmoid_derivative(predicted_output)

 error_hidden_layer3 = d_predicted_output.dot(output_weights.T)
 d_hidden_layer3 = error_hidden_layer3 * sigmoid_derivative(hidden_layer_output3)

 error_hidden_layer2 = d_hidden_layer3.dot(hidden_weights3.T)
 d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer_output2)

 error_hidden_layer1 = d_hidden_layer2.dot(hidden_weights2.T)
 d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer_output1)

 #error_hidden_layer2 = d_hidden_layer3.dot(hidden_weights3.T)
 #d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer_output2)

 #error_hidden_layer1 = d_hidden_layer2.dot(hidden_weights2.T)
 #d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer_output1)


	#Updating Weights and Biases
 output_weights += hidden_layer_output3.T.dot(d_predicted_output) * lr
 output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr

 hidden_weights3 += hidden_layer_output2.T.dot(d_hidden_layer3) * lr
 hidden_bias3 += np.sum(d_hidden_layer3,axis=0,keepdims=True) * lr

 hidden_weights2 += hidden_layer_output1.T.dot(d_hidden_layer2) * lr
 hidden_bias2 += np.sum(d_hidden_layer2,axis=0,keepdims=True) * lr

 #hidden_weights3 += hidden_layer_output2.T.dot(d_hidden_layer3) * lr
 #hidden_bias3 += np.sum(d_hidden_layer3,axis=0,keepdims=True) * lr

 #hidden_weights2 += hidden_layer_output1.T.dot(d_hidden_layer2) * lr
 #hidden_bias2 += np.sum(d_hidden_layer2,axis=0,keepdims=True) * lr

 hidden_weights1 += inputs.T.dot(d_hidden_layer1) * lr
 hidden_bias1 += np.sum(d_hidden_layer1,axis=0,keepdims=True) * lr

print("\nFinal hidden weights1: ",end='')
print(hidden_weights1)
print("Final hidden bias1: ",end='')
print(hidden_bias1)
print("Final hidden weights2: ",end='')
print(hidden_weights2)
print("Final hidden bias2: ",end='')
print(hidden_bias2)
print("Final hidden weights3: ",end='')
print(hidden_weights3)
print("Final hidden bias3: ",end='')
print(hidden_bias3)
print("Final output weights: ",end='')
print(*output_weights)
print("Final output bias: ",end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predicted_output)

print("\nnode 1: ",end='')
print(*hidden_layer_output1)

print("\nnode 2: ",end='')
print(hidden_layer_output2)

print("\nnode 3: ",end='')
print(*hidden_layer_output3)

for _ in range(1):
	#Forward Propagation
 hidden_layer_activation1 = np.dot(inputs,hidden_weights1)
 hidden_layer_activation1 += hidden_bias1
 hidden_layer_output1 = sigmoid(hidden_layer_activation1)

 #hidden_layer_activation2 = np.dot(hidden_layer_output1,hidden_weights2)
 #hidden_layer_activation2 += hidden_bias2
 #hidden_layer_output2 = sigmoid(hidden_layer_activation2)

 #hidden_layer_activation3 = np.dot(hidden_layer_output1,hidden_weights3)
 #hidden_layer_activation3 += hidden_bias3
 #hidden_layer_output3 = sigmoid(hidden_layer_activation3)

 hidden_layer_activation2 = np.dot(hidden_layer_output1,hidden_weights2)
 hidden_layer_activation2 += hidden_bias2
 hidden_layer_output2 = sigmoid(hidden_layer_activation2)

 hidden_layer_activation3 = np.dot(hidden_layer_output2,hidden_weights3)
 hidden_layer_activation3 += hidden_bias3
 hidden_layer_output3 = sigmoid(hidden_layer_activation3)

 output_layer_activation = np.dot(hidden_layer_output3,output_weights)
 output_layer_activation += output_bias
 predicted_output = sigmoid(output_layer_activation)


 an_array1 = np.where(hidden_layer_output1 < 0.1, 0, hidden_layer_output1)
 print(an_array1)

 an_array2 = np.where(hidden_layer_output2 < 0.1, 0, hidden_layer_output2)
 print(an_array2)

 an_array3 = np.where(hidden_layer_output3 < 0.1, 0, hidden_layer_output3)
 print(an_array3)

 for i in range hidden_layer_output1:
   for j in range hidden_layer_output1:
     if




print("\nnode 1: ",end='')
print(hidden_layer_output1)

print("\nnode 2: ",end='')
print(hidden_layer_output2)

print("\nnode 3: ",end='')
print(*hidden_layer_output3)

print("\ninput: ",end='')
print(inputs)
