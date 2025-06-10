from My_engine import Value
from neural_net import MLP
import math
import random
import matplotlib.pyplot as plt
import numpy as np


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
print(n(x))

