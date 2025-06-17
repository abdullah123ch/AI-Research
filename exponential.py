import math
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
            p.gradient = 0.0
        loss.backward()

        # Gradient descent
        learning_rate = 0.001
        for p in model.parameters():
            p.data -= learning_rate * p.gradient

        print(f"Epoch {k}, Loss: {loss.data:.4f}")

    return loss

inputs = [[(x/10)] for x in range(-20, 21)]
targets = [math.exp(x[0]) for x in inputs]

n = MLP(1, [6, 6, 1])  # 1 input, 2 hidden layers with 4 neurons each, and 1 output neuron

# Train the model using the new train function
loss = train(n, inputs, targets, steps=500, learningRate=0.1)

print("")
print("Final parameters:")  
print("")
for x, ygt in zip(inputs, targets):
    ypred = n(x)
    print(f"Input: {x[0]:.2f}, Predicted: {ypred.data:.2f}, Ground Truth: {ygt.data:.2f}")


# display the final of the model
draw_dot(loss).render('cubic', view=True)  # This will create a file named 'quadratic.svg' and open it

