# Optimization Algorithms in C for Binary Classification

## Overview
This repository features implementations of three pivotal optimization algorithms in C, tailored for binary classification tasks. The code reads a dataset where each line has a label of either 1 or -1, processes this data into one-hot vectors, and applies the following optimization algorithms:

### Algorithms

#### 1. Gradient Descent (GD)
A first-order iterative optimization algorithm, Gradient Descent is utilized to find the minimum of a function. It is a cornerstone in machine learning for optimizing loss functions.

#### 2. Stochastic Gradient Descent (SGD)
As a variant of Gradient Descent, SGD differs by updating model parameters using just one or a few training examples per iteration. This approach lends itself to greater efficiency with large datasets.

#### 3. Adaptive Moment Estimation (ADAM)
ADAM merges concepts from RMSProp and SGD with momentum. It adapts learning rates for each parameter, making it highly effective for scenarios with extensive data and numerous parameters.

### Getting Started

To use these implementations:

1. **Run the C Program**: 
   - Simply execute the `main.c` program. 
   - The program automatically creates folders containing the result files.

2. **Visualize Results**:
   - After the C program completes, run the accompanying Python script to visualize the results.
