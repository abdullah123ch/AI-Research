import math

class Value:
    def __init__(self, data, children=(), operation='', label=''):
        self.data = data
        self.gradient = 0
        self.backpath = lambda: None
        self._prev = set(children)
        self.operation = operation
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def backpath():
            self.gradient += 1.0 * out.gradient
            other.gradient += 1.0 * out.gradient
        out.backpath = backpath
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def backpath():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out.backpath = backpath
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def backpath():
            self.gradient += other * (self.data ** (other - 1)) * out.gradient
        out.backpath = backpath

        return out
    
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def backpath():
            self.gradient += (1 - t**2) * out.gradient
        out.backpath = backpath
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def backpath():
            self.gradient += (1 - t**2) * out.gradient
        out.backpath = backpath
        
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def backpath():
            self.gradient += s * (1 - s) * out.gradient
        out.backpath = backpath

        return out
    
    def backward(self):    
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.gradient = 1.0
        for node in reversed(topo):
            node.backpath()

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
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # Backward pass
        for p in model.parameters():
            p.gradient = 0.0
        loss.backward()

        # Update
        for p in model.parameters():
            p.data += -learningRate * p.gradient
            
        print(f"Step: {k}, Loss: {loss.data:.8f}")
    
    return loss