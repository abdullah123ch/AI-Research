# AutoGrad: Minimal Neural Network Engine with Visualization

AutoGrad is a minimal neural network engine with automatic differentiation, inspired by PyTorch. It provides tools for building, training, and visualizing neural networks from scratch, making it ideal for learning and experimentation.

---

## Features

- **Automatic Differentiation:** Core engine for forward and backward passes.
- **Neural Network Module:** Easily define and train multi-layer perceptrons (MLPs).
- **Graph Visualization:** Visualize computation graphs using Graphviz.
- **Examples:** Ready-to-run example demonstrating training and visualization.
- **PyTorch Comparison:** Framework for comparing AutoGrad with PyTorch.

---

## Project Structure
- **AutoGrad/ display.py:**  Visualization utilities for computation graphs 
- **My_engine.py:**  Core engine for autograd and training 
- **neural_net.py:** Neural network (MLP) implementation 
- **pycache:** Python cache files 
- **AutoGrad vs Pytorch.py** Script for comparing with PyTorch  
- **Example.py** Example usage and training script 
- **example.svg** # Example output graph 
- **README.md**  Project documentation

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Graphviz](https://graphviz.gitlab.io/download/) (for visualization)
- Python package: `graphviz`

### Installation

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd <repo-directory>
    ```

2. **Install Python dependencies:**
    ```sh
    pip install graphviz
    ```

3. **Install Graphviz system package:**
    - On Ubuntu: `sudo apt-get install graphviz`
    - On Windows: [Download Graphviz](https://graphviz.gitlab.io/download/)

---

## Custom Usage
- **Define neural networks:** Use the MLP class in AutoGrad/neural_net.py.
- **Train models:** Use the train function in AutoGrad/My_engine.py.
- **Visualize graphs:** Use the draw_dot function in AutoGrad/display.py.

---
## License
This project is for educational use.

---

Inspired by micrograd and PyTorch.
