import random
from My_engine import Value


class Neuron:
    def __init__(self, inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.w), "Input size must match weights size"
        activation = self.b
        for w_i, x_i in zip(self.w, x):
            activation = activation + (w_i * x_i)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, inputs, nout):
        self.neurons = [Neuron(inputs) for _ in range(nout)]

    def __call__(self, x):
        assert len(x) == len(self.neurons[0].w), "Input size must match weights size"
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params
class MLP:
    def __init__(self, inputs, nouts):
        size = [inputs] + nouts
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        assert len(x) == len(self.layers[0].neurons[0].w), "Input size must match weights size"
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    

def loss(Ygt, Ypred):
        assert len(Ygt) == len(Ypred), "Ground truth and prediction must have the same length"
        loss = sum((y_gt - y_pred) ** 2 for y_gt, y_pred in zip(Ygt, Ypred)) 
        return loss
        