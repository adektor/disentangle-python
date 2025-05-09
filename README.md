# Tensor Disentangler

This repository provides functionality for reducing entanglement in a tensor across specified dimensions by applying a unitary transformation to another set of dimensions. Such disentangling is important for many tensor network algorithms.

## Main Function

### `disentangle(X, dis_dims, svd_dims, **kwargs)`

- `X`: The input tensor to be disentangled.
- `dis_dims`: A list of dimensions of `X` on which the unitary disentangling matrix acts.
- `svd_dims`: A list of dimensions of `X` across which the entanglement is minimized.

For example, if `X` is a 4D NumPy array with dimensions `[0, 1, 2, 3]`, then

```python
disentangle(X, dis_dims=[0, 1], svd_dims=[0, 2], **kwargs)
```
optimizes a unitary matrix `Q` to minimize the error of the truncated SVD in the following tensor network diagram. 

![Diagram](images/dis_4ten.svg)

The user can specify additional keyword arguments, such as:
- Optimization algorithm
- Maximum wall time for disentangling
- Initial disentangler guess
- Other advanced options
