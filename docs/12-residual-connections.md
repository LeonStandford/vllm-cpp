# ➕ Residual connections

It's a simple technique used in many different deep learning models, where you add inputs to the outputs, so your input data is never fully gone. It's done for stability in training. See [more info here](https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55/). From our perspective it's just adding two same sized vectors, elementwise.

```cpp
__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
}

// (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
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

[← RoPE](11-rope.md) · [🏠 Index](../README.md) · [cublasGemmEx →](13-cublasgemmex.md)
