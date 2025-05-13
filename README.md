# Tensor Disentangler

Disentangling is important for many tensor network algorithms. This repository provides functionality for optimizing unitary disentangler matrices to reduce entanglement across specified dimensions of a input tensor. The user provides the tensor, dimensions on which the unitary matrix is applied, and dimensions across which the entanglement is minimized. 

## Usage

### `disentangle(X, dis_dims, svd_dims, **kwargs)`

- `X`: The input tensor (NumPy array) to be disentangled.
- `dis_dims`: A list of dimensions of `X` on which the unitary disentangling matrix acts.
- `svd_dims`: A list of dimensions of `X` across which the entanglement is minimized.
- `chi`: target truncation rank

For example, if `X` is a 4D NumPy array with dimensions `[0, 1, 2, 3]`, then

```python
Q = disentangle(X, dis_dims=[0, 1], svd_dims=[0, 2], chi=4, **kwargs)
```
optimizes a unitary matrix `Q` to minimize the error of the rank-`chi` truncated SVD in the following tensor network diagram. 

<img src="images/dis_4ten.svg" alt="Disentangling Diagram" width="400"/>

The user can specify additional keyword arguments, such as:
- Optimization algorithm
- Maximum wall time for disentangling
- Initial disentangler guess
- Other advanced options
