from AutoGrad.neural_net import MLP
from AutoGrad.My_engine import Value
from AutoGrad.display import draw_dot

def train(model, xs, ys, steps=2000, learningRate=0.1):
    """
    Trains the given model on the provided data.

    Args:
        model: The neural network model (must have parameters(), __call__(), etc.).
        xs: List of input samples.
        ys: List of ground truth outputs.
        steps: Number of training iterations.
        learningRate: Learning rate.
    """
    for k in range(steps):
        # Forward pass
        predictions = [model(x) for x in inputs]
        
        # Compute MSE Loss
        loss = sum((y_pred - y_target)**2 for y_pred, y_target in zip(predictions, targets)) / len(inputs)

        # Zero gradients, backward pass
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # Gradient descent
        learning_rate = 0.001
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        print(f"Epoch {k}, Loss: {loss.data:.4f}")

    return loss

def f(x):
    return 2 * x**2 + 3 * x + 1

import random

def generate_data():
    inputs = []
    targets = []
    x = random.uniform(-10, 10)
    y = f(x)
    inputs.append([Value(x)])  # list of one Value object
    targets.append(Value(y))   # single target Value
    return inputs, targets

inputs, targets = generate_data()

n = MLP(1, [16, 16, 1])  # 1 input, 2 hidden layers with 4 neurons each, and 1 output neuron

# Train the model using the new train function
loss = train(n, inputs, targets, steps=100, learningRate=0.1)

print("")
print("Final parameters:")  
print("")
for x, ygt in zip(inputs, targets):
    ypred = n(x)
    print(f"Input: {x[0].data}, Predicted: {ypred.data}, Ground Truth: {ygt.data}")

# display the final of the model
# draw_dot(loss).render('quadratic', view=True)  # This will create a file named 'quadratic.svg' and open it

