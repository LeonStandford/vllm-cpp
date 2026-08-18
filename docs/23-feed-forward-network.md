# 🕸️ Feed forward network

A [feed-forward network](https://en.wikipedia.org/wiki/Feedforward_neural_network) / fully connected network / multi-layer perceptron. Three linear layers and SiLU.

```cpp
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              HIDDEN_DIM,       // m
              1,                // n
              EMBEDDING_LENGTH, // k
              &gate_alpha,
              weights.mlp_gate_proj[layer],
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              rms_norms,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              &gate_beta,
              gate,
              CUDA_R_16BF,
              HIDDEN_DIM,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);

// (1, 2048) * (2048, 8192) -> (1, 8192)
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              HIDDEN_DIM,       // m
              1,                // n
              EMBEDDING_LENGTH, // k
              &up_alpha,
              weights.mlp_up_proj[layer],
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              rms_norms,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              &up_beta,
              up,
              CUDA_R_16BF,
              HIDDEN_DIM,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);

silu(gate, up, 1);

down = buf_2048_2;
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              EMBEDDING_LENGTH, // m
              1,                // n
              HIDDEN_DIM,       // k
              &down_alpha,
              weights.mlp_down_proj[layer],
              CUDA_R_16BF,
              HIDDEN_DIM,
              gate,
              CUDA_R_16BF,
              HIDDEN_DIM,
              &down_beta,
              down,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);
```

---

[← Argmax](22-argmax.md) · [🏠 Index](../README.md) · [Buffer reuse →](24-buffer-reuse.md)
