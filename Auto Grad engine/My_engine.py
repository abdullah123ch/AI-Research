import math
import random
import matplotlib.pyplot as plt
import numpy as np


class Value:
    def __init__(self, data, children = (), operator='', label=''):
        self.data = data
        self.gradient = 0.0
        self.backward = lambda: None
        self.previous = set(children)
        self.operator = operator
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data})"
        
    def add(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def backward():
            self.gradient += 1.0 * out.gradient
            other.gradient += 1.0 * out.gradient
        out.backward = backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
    
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = backward
        return out
    
    def rmultiply(self, other):  # other * self
        return self.multiply(other)
    
    def divide(self, other):  # self / other
        return self.multiply(other ** -1)
    
    def negate(self):  # -self
        return self.multiply(-1)
    
    def subtract(self, other):  # self - other
        return self.add(other.negate())
    
    def radd(self, other):  # other + self
        return self.add(other)

    def power(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def backward():
            self.gradient += other * (self.data ** (other - 1)) * out.gradient
        out.backward = backward

        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def backward():
            self.gradient += out.data * out.gradient
        out.backward = backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def backward():
            self.gradient += (1 - t ** 2) * out.gradient
        out.backward = backward

        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def backward():
            self.gradient += s * (1 - s) * out.gradient
        out.backward = backward

        return out
    
    def true_divide(self, other):
        return self.multiply(other ** -1)
    
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
            
        self.grad = 1.0
        for node in reversed(topo):
            node.backward()