# Optimization Algorithms in C

This repository contains implementations of three key optimization algorithms written in C. These algorithms are fundamental for various machine learning and data analysis applications. The code is structured in `main.c` and showcases the following algorithms:

### 1. GD (Gradient Descent)
Gradient Descent is a first-order iterative optimization algorithm used to find the minimum of a function. It's widely used in machine learning to optimize loss functions.

### 2. SGD (Stochastic Gradient Descent)
Stochastic Gradient Descent is a variant of Gradient Descent. Unlike GD, which uses the entire data set to update the model parameters in each iteration, SGD updates the parameters using only one or a few training examples. This makes SGD faster and more suitable for large datasets.

### 3. ADAM (Adaptive Moment Estimation)
ADAM is an optimization algorithm that combines ideas from RMSProp and SGD with momentum. It computes adaptive learning rates for each parameter. ADAM is particularly effective in settings with large amounts of data and parameters.

## Project Overview

This project aims to provide a clear and efficient implementation of these optimization algorithms in C. It's designed for educational purposes, helping students and enthusiasts understand the inner workings of these algorithms.

- **Language Used**: C
- **Main File**: `main.c`
- **Algorithms Implemented**:
  - Gradient Descent (GD)
  - Stochastic Gradient Descent (SGD)
  - Adaptive Moment Estimation (ADAM)

## Getting Started

To get started with this project:
1. Clone the repository.
2. Navigate to the project directory.
3. Compile the `main.c` file.
4. Run the executable to see the algorithms in action.

## Contribution

Contributions are welcome! If you have suggestions or want to improve the implementations, feel free to fork the repository and submit a pull request.
