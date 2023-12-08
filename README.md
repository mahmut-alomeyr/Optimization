# Optimization Algorithms in C

This repository contains implementations of three key optimization algorithms written in C. These algorithms are fundamental for various machine learning and data analysis applications. The code is structured in `main.c` and showcases the following algorithms:

### 1. GD (Gradient Descent)
Gradient Descent is a first-order iterative optimization algorithm used to find the minimum of a function. It's widely used in machine learning to optimize loss functions.

### 2. SGD (Stochastic Gradient Descent)
Stochastic Gradient Descent is a variant of Gradient Descent. Unlike GD, which uses the entire data set to update the model parameters in each iteration, SGD updates the parameters using only one or a few training examples. This makes SGD faster and more suitable for large datasets.

### 3. ADAM (Adaptive Moment Estimation)
ADAM is an optimization algorithm that combines ideas from RMSProp and SGD with momentum. It computes adaptive learning rates for each parameter. ADAM is particularly effective in settings with large amounts of data and parameters.

### Getting Started

To use these implementations:

1. **Run the C Program**: 
   - Simply execute the `main.c` program. 
   - The program automatically creates folders containing the result files.

2. **Visualize Results**:
   - After the C program completes, run the accompanying Python script to visualize the results.
