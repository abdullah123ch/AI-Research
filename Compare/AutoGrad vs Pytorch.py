from AutoGrad.My_engine import Value, train
import AutoGrad.neural_net as nn
from AutoGrad.display import draw_dot
import torch

# Create a simple neural network using AutoGrad
model = nn.MLP(2, [4, 3, 1])

# Create some dummy input data
x = [Value(0.5), Value(-0.5)]

Loss = train(model, [x], [Value(1.0)], steps=1000, learningRate=0.01)

# Visualize the computation graph
draw_dot(Loss).render('autograd_model', format='png')