# Feedforward Neural Network and Backpropagation
This is a simple Feedforward Neural Network(FNN), which is specifically a 3-layer feedforward network built from scratch using only Numpy library. The network has:
1. 2 Input neurons
2. 1 hidden layer with 3 neurons.
3. 3 output neurons.
In this, I have done forward pass and then backward pass for backpropagation. I have used the sigmoid activation function and trained the network using backpropagation to minimize the total error between predicted outputs and target values.
## How it Works
### Forward Pass:
i. Hidden Layer Inputs: h=Wx
ii. Hidden Layer Outputs(using Sigmoid activation funcction): Oh=σ(h)= 1/1+e^-h
iii.Output layer inputs:𝑜=Θ⋅𝑂ℎ
iv. Final outputs: 𝑂𝑗=𝜎(𝑜)
Loss Function: Mean Squared Error (MSE): 𝐿=1/2∑(𝑡−𝑂𝑗)2 where t=target values and Oj=predicted outputs.
### Backpropagation:
