# Tensor Disentangler

tensor-disentangler is a Python package that optimizes 'disentangling' unitary matrices to reduce bond-dimension in a given tensor. It is built on Pymanopt and designed to be used in tensor network algorithms for quantum many-body calculations and beyond. For example, disentangling is important for tensor network renormalization, isometric tensor network states, MERA, purified mixed-state MPS, and unitary tensor networks. 

The user provides the tensor, dimensions of the tensor on which the unitary matrix is applied, and dimensions across which entanglement is minimized. 

## Installation 

We should make this available via pip

## Usage

### `disentangle(X, dis_legs, svd_legs, **kwargs)`

- `X`: The input tensor (NumPy array) to be disentangled.
- `dis_legs`: A list of legs of `X` on which the unitary disentangling matrix acts.
- `svd_legs`: A list of legs of `X` across which the entanglement is minimized.

For example, if `X` is a 4D NumPy array then

```python
Q, U, S, V = disentangle(X, dis_legs=[0, 1], svd_legs=[0, 2], **kwargs)
```
optimizes a unitary matrix `Q` to minimize the error when truncating the SVD in the following tensor network diagram

<img src="images/dis_4ten.svg" alt="Disentangling Diagram" width="400"/>

## Features
tensor-disentangler has support for various optimizers and objective functions for disentangling. 
The optimizer, objective function, and other options can be specified with keyword arguments. For a comprehensive list we should write some documentation. 

- Optimization algorithm: Alternating, Riemannian Conjugate Gradient, Riemannian Steepest Descent, and more
- Optimization objective function: Renyi entropy, Von-Neumann entropy, or truncation error
- Initial disentangler: identity, random, or user-supplied
- Maximum wall time and other optimizer specific stopping criteria
- Other advanced options
  
