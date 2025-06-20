from AutoGrad.neural_net import MLP
from AutoGrad.display import draw_dot

def train(model, inputs, targets, steps=2000, learningRate=0.1):
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

        if k % 100 == 0:
            print(f"Epoch {k}, Loss: {loss.data:.4f}")

    return loss

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
] # input samples
ys = [1.0, -1.0, -1.0, 1.0] # desired outputs
n = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers with 4 neurons each, and 1 output neuron

# Train the model using the new train function
loss = train(n, xs, ys, steps=2000, learningRate=0.1)

print("")
print("Final parameters:")
print("")
for x, ygt in zip(xs, ys):
    ypred = n(x)
    print(f"Input: {x}, Predicted: {ypred.data}, Ground Truth: {ygt}")# display the final of the model
# draw_dot(loss).render('example', view=True)  # This will create a file named 'example.svg' and open it

