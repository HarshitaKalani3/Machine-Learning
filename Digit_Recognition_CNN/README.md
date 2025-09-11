# Digit_Recognition_CNN(Mnist + LeNet-5)
It implements a CNN(Convolutional Neural Network)(LeNet-5) to recognize handwritten digits(0-9) using the MNIST dataset. It also supports custom image inference from a given image, confusion matrix and one-hot encoding demonstration of some digits which are to be used for calculations.
## Tech Stack
1. Python3, PyTorch, Torchvision
2. scikit-learn, Pillow, Requests
3. Pandas, Numpy, Matplotlib
## How It Works
1. It first loads and preprocess MNIST dataset given.
2. It then trains a LeNet-5 CNN with Adam optimizer.
3. Validation of the model and at each step saving the best accuracy and updating it.
4. Then, generate confusion matrix for better predictions.
5. At last, run inference on external maybe from web handwritten digit images.
## Example Output
 Epoch:1, Accuracy:98.32%
 Epoch:2, Accuracy:98.74%
 Predicted:2, Probability: 99.14%
 Manual dig: 7
 Sum: 9
## Results
1. ~98-99% accuracy on MNIST data set.
2. Correct predictions on external handwritten digits.
3. Correct summation of values from digit recognition from image as well as from one-hot encoded value.
## Under Guidance of: Ritambhra Korpal Ma'am
## Author
Harshita Kalani (https://github.com/HarshitaKalani3)
