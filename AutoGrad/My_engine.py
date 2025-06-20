class Value:
    def __init__(self, data, _children=(), operation=''):
        self.data = data
        self.gradient = 0
        # internal variables used for autograd graph construction
        self.back = lambda: None
        self._prev = set(_children)
        self.operation = operation # the op that produced this node, for graphviz / debugging / etc
        self.label = f"{self.operation}({', '.join(str(v.data) for v in _children)})" if _children else str(data)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def back():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out.back = back

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def back():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out.back = back

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def back():
            self.gradient += (other * self.data**(other-1)) * out.gradient
        out.back = back

        return out

    def exp(self):
        out = Value(2.718**self.data, (self,), 'exp')

        def back():
            self.gradient += out.data * out.gradient
        out.back = back

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def back():
            self.gradient += (out.data > 0) * out.gradient
        out.back = back

        return out
     
    def tanh(self):
        out = Value((2 / (1 + 2.718**(-2 * self.data))) - 1, (self,), 'tanh')

        def back():
            self.gradient += (1 - out.data**2) * out.gradient
        out.back = back

        return out
    
    def sigmoid(self):
        out = Value(1 / (1 + 2.718**(-self.data)), (self,), 'sigmoid')

        def back():
            self.gradient += out.data * (1 - out.data) * out.gradient
        out.back = back

        return out
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.gradient = 1
        for v in reversed(topo):
            v.back()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"