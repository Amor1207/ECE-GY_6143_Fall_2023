import numpy as np
# Define the sigmoid activation function again
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Given weights and biases from the neural network diagram
weights = {
    'x1_h1': 2.5,
    'x2_h1': 3,
    'x1_h2': -2.5,
    'x2_h2': -3,
    'xb_h1': 1.5,  # Bias for h1
    'xb_h2': 1,   # Bias for h2
    'h1_o1': 1,
    'h2_o1': 0.5,
    'hb_o1': -2  # Bias for o1
}

# Input values
x1 = -1
x2 = 2

# Calculate the input for each hidden unit
h1_input = weights['x1_h1'] * x1 + weights['x2_h1'] * x2 + weights['xb_h1']
h2_input = weights['x1_h2'] * x1 + weights['x2_h2'] * x2 + weights['xb_h2']

# Apply sigmoid activation function to each hidden unit's input
h1_output = sigmoid(h1_input)
h2_output = sigmoid(h2_input)

# Calculate the input for the output unit
o1_input = weights['h1_o1'] * h1_output + weights['h2_o1'] * h2_output + weights['hb_o1']

# Apply sigmoid activation function to the output unit's input
#sigmoid输出激活层
#o1_output = sigmoid(o1_input)
#线性输出激活层
o1_output = o1_input
print(h1_output,h2_output,o1_output)

# ------------------------------------------

#若输出节点为sigmoid激活层，则反向传播误差
# delta_o1 = o1_output * (1 - o1_output) * (o1_output-1)
# print(delta_o1)
# delta_h1 = delta_o1 * weights['h1_o1'] * h1_output * (1 - h1_output)
# print(delta_h1)
# delta_h2 = delta_o1 * weights['h2_o1'] * h2_output * (1 - h2_output)
# print(delta_h2)

#若输出节点为线性激活层，则反向传播误差
# The true value is given as y = 1
y_true = 1
# Compute the backpropagation error for o1 (since we have a linear activation at the output, the derivative is 1)
delta_o1 = o1_output - y_true
# Compute the derivative of the sigmoid function for h1
sigmoid_derivative_h1 = h1_output * (1 - h1_output)
# Compute the backpropagation error for h1 and h2
delta_h1 = delta_o1 * weights['h1_o1'] * sigmoid_derivative_h1
sigmoid_derivative_h2 = h2_output * (1 - h2_output)
delta_h2 = delta_o1 * weights['h2_o1'] * sigmoid_derivative_h2
print("delta_o1:",delta_o1,"delta_h1:",delta_h1,"delta_h2:",delta_h2)

# ------------------------------------------
# Compute gradients of loss function with respect to weights
# Calculating the gradient for the weights from input to h1
grad_w_x1_h1 = delta_h1 * x1
grad_w_x2_h1 = delta_h1 * x2
grad_w_xb_h1 = delta_h1 * 1

# Calculating the gradient for the weights from input to h2
grad_w_x1_h2 = delta_h2 * x1
grad_w_x2_h2 = delta_h2 * x2
grad_w_xb_h2 = delta_h2 * 1

# For the output neuron weights:
grad_w_h1_o1 = delta_o1 * h1_output
grad_w_h2_o1 = delta_o1 * h2_output
grad_w_hb_o1 = delta_o1 * 1

# The gradients for W_o1 W_h1 and W_h2
grad_W_o1 = np.array([grad_w_h1_o1, grad_w_h2_o1, grad_w_hb_o1])
grad_W_h1 = np.array([grad_w_x1_h1, grad_w_x2_h1, grad_w_xb_h1])
grad_W_h2 = np.array([grad_w_x1_h2, grad_w_x2_h2, grad_w_xb_h2])

print(grad_W_o1,grad_W_h1,grad_W_h2)

# ------------------------------------------
# Update weights
# Compute the updated weights for both the hidden layer and the output layer by performing one step of gradient descent. Use a learning rate of 0.3.

# Learning rate
learning_rate = 0.3
updated_weights = {
    'h1_o1': weights['h1_o1'] - learning_rate * grad_w_h1_o1,
    'h2_o1': weights['h2_o1'] - learning_rate * grad_w_h2_o1,
    'hb_o1': weights['hb_o1'] - learning_rate * grad_w_hb_o1,
    'x1_h1': weights['x1_h1'] - learning_rate * grad_w_x1_h1,
    'x2_h1': weights['x2_h1'] - learning_rate * grad_w_x2_h1,
    'xb_h1': weights['xb_h1'] - learning_rate * grad_w_xb_h1,
    'x1_h2': weights['x1_h2'] - learning_rate * grad_w_x1_h2,
    'x2_h2': weights['x2_h2'] - learning_rate * grad_w_x2_h2,
    'xb_h2': weights['xb_h2'] - learning_rate * grad_w_xb_h2
}

updated_W_o1 = [updated_weights['h1_o1'], updated_weights['h2_o1'], updated_weights['hb_o1']]
updated_W_h1 = [updated_weights['x1_h1'], updated_weights['x2_h1'], updated_weights['xb_h1']]
updated_W_h2 = [updated_weights['x1_h2'], updated_weights['x2_h2'], updated_weights['xb_h2']]
print(updated_W_o1,updated_W_h1,updated_W_h2)