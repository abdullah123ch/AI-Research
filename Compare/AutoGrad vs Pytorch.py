from AutoGrad.My_engine import Value, train
from AutoGrad.neural_net import NeuralNet
from AutoGrad.display import draw_dot
import torch

def main():
    # Define the neural network using AutoGrad
    net = NeuralNet([
        Value(2, name='x1'),
        Value(3, name='x2'),
        Value(4, name='x3'),
        Value(5, name='x4'),
        Value(6, name='x5')
    ])

    # Train the network
    train(net)

    # Draw the computation graph
    draw_dot(net)

    # Print the final values of the inputs
    print("Final values:")
    for i in range(1, 6):
        print(f"x{i}: {getattr(net, f'x{i}').data}")