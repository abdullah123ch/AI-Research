from AutoGrad.neural_net import MLP
from AutoGrad.My_engine import train
from AutoGrad.display import draw_dot

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired outputs
n = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers with 4 neurons each, and 1 output neuron

# Train the model using the new train function
loss = train(n, xs, ys, steps=2000, learningRate=0.1)

print("")
print("Final parameters:")
print("")
for x, ygt in zip(xs, ys):
    ypred = n(x)
    print(f"Input: {x}, Predicted: {ypred.data}, Ground Truth: {ygt}")

# display the final of the model
draw_dot(loss).render('model', view=True)  # This will create a file named 'model.svg' and open it

