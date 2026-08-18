# ✖️ cublasGemmEx

[Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) is one of the main operations used in deep learning, most notably in large language models. Matrix is a table of numbers. It has rows and columns. A single row and a single column is called a vector -- a sequence of numbers. Matrix multiplication uses two matrices, A and B, as an input and produces matrix C as an output. Matrix A has dimensions (M, K). Matrix B has dimensions (K, N). Both matrix A and B share the same dimension K. In other words, rows of matrix A have the same length as columns of matrix B. When you multiply A by B, you get a new matrix C with dimensions (M, N):

$$A (M,K) \times B(K,N)=C(M,N)$$

Every element of matrix C ($c_{ij}$) is a [dot product](https://en.wikipedia.org/wiki/Dot_product) of i-th row of A and j-th column of B. Dot product is a sum of all pairs, where each pair is a result of multiplying numbers from i-th row of A with numbers from j-th row of B on the same indices within their vectors (both row and column are vectors):

$$c_{ij} = a_{i0}b_{0j} + a_{i1}b_{1j}+...+a_{ik}b_{kj}=\sum_{x=0}^{k} a_{ix}b_{xj}$$

Back to large language models. Matrix multiplication happens when computing attention and Q, K and V projections. Most popular hardware to compute it efficiently are NVIDIA GPUs. They provide an important library, [cuBLAS](https://developer.nvidia.com/cublas), which allows you to run high performance linear algebra computations on their GPUs, including matrix multiplication, using [cublasGemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex) function.

---

[← Residual connections](12-residual-connections.md) · [🏠 Index](../README.md) · [The column-major to row-major transposition trick →](14-the-column-major-to-row-major-transposition-trick.md)
