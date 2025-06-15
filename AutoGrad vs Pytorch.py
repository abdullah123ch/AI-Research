from AutoGrad.My_engine import Value
from AutoGrad.neural_net import MLP, Layer, Neuron
from AutoGrad.display import draw_dot
import torch


inputs = [[1.0], [2.0], [3.0], [4.0]]
targets = [3.0, 5.0, 7.0, 9.0]
# AutoGrad
model = MLP(1, [4, 4, 1])  # 1 input, 2 hidden layers with 2 neurons each, and 1 output neuron
loss = train(model, inputs, targets, steps=2000, learningRate=0.01)
print("\nFinal parameters:")
for x, ygt in zip(inputs, targets):
    ypred = model(x)
    print(f"Input: {x}, Predicted: {ypred.data}, Ground Truth: {ygt}")
# display the final of the model
# draw_dot(loss).render('model', view=True)  # This will create a file named 'model.svg' and open it
