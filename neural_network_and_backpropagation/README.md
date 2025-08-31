# Feedforward Neural Network and Backpropagation
This is a simple Feedforward Neural Network(FNN), which is specifically a 3-layer feedforward network built from scratch using only Numpy library. The network has:
1. 2 Input neurons
2. 1 hidden layer with 3 neurons.
3. 3 output neurons.
In this, I have done forward pass and then backward pass for backpropagation. I have used the sigmoid activation function and trained the network using backpropagation to minimize the total error between predicted outputs and target values.
## How it Works
### Forward Pass:
1. Hidden Layer Inputs: h = Wx
2. Hidden Layer Outputs(using Sigmoid activation funcction): Oh = σ(h) = 1/1+e^-h
3. Output layer inputs: o = Θ⋅𝑂h
4. Final outputs: Oj = 𝜎(o)
5. Loss Function: Mean Squared Error (MSE): L = 1/2∑(t-Oj)2 where t=target values and Oj=predicted outputs.
### Backpropagation:
1. Error at output layer: 𝛿out = (Oj-t)⋅𝜎′(Oj)
2. Error at hidden layer: 𝛿hidden = Θ𝑇⋅𝛿out⋅𝜎′(Oh)
3. Gradient for weights (hidden to output): ΔΘ = 𝛿out⊗𝑂h
4. Gradient for weights (input to hidden): Δ𝑊 = 𝛿hidden⊗x
