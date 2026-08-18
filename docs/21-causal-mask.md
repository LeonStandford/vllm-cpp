# 🎭 Causal mask

Every token can attend only to previous tokens. See [this good and straighforward explanation](https://outcomeschool.com/blog/causal-masking-in-attention), so you can code it by yourself. It's really helpful to see the diagrams.

< TODO write more >

```cpp
__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int column = threadIdx.x;
    int row = blockIdx.x % num_tokens;
    if (column > row)
    {
        input[blockIdx.x * num_tokens + threadIdx.x] = -HUGE_VALF;
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

---

[← Softmax](20-softmax.md) · [🏠 Index](../README.md) · [Argmax →](22-argmax.md)
