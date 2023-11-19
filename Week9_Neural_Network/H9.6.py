import numpy as np
# Define the sigmoid activation function again
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Given weights and biases from the neural network diagram
weights = {
    'x1_h1': 2.5,
    'x2_h1': 3,
    'x1_h2': -0.5,
    'x2_h2': -3,
    'xb_h1': 0.5,  # Bias for h1
    'xb_h2': 1,   # Bias for h2
    'h1_o1': 2,
    'h2_o1': -1,
    'hb_o1': -1  # Bias for o1
}

# Input values
x1 = 0
x2 = 1

# Calculate the input for each hidden unit
h1_input = weights['x1_h1'] * x1 + weights['x2_h1'] * x2 + weights['xb_h1']
h2_input = weights['x1_h2'] * x1 + weights['x2_h2'] * x2 + weights['xb_h2']

# Apply sigmoid activation function to each hidden unit's input
h1_output = sigmoid(h1_input)
h2_output = sigmoid(h2_input)

# Calculate the input for the output unit
o1_input = weights['h1_o1'] * h1_output + weights['h2_o1'] * h2_output + weights['hb_o1']

# Apply sigmoid activation function to the output unit's input
o1_output = sigmoid(o1_input)
print(h1_output,h2_output,o1_output)

delta_o1 = o1_output * (1 - o1_output) * (o1_output-1)
print(delta_o1)
delta_h1 = delta_o1 * weights['h1_o1'] * h1_output * (1 - h1_output)
print(delta_h1)
delta_h2 = delta_o1 * weights['h2_o1'] * h2_output * (1 - h2_output)
print(delta_h2)
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