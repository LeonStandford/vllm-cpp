# 〰️ SiLU

[SiLU](https://arxiv.org/pdf/1702.03118) is an activation function used in our reference LLM. It introduces "non-linearity" into a model. It means that weights can be zeroed when they are not needed. It helps in training models. Almost all machine learning models have them in their architecture, even the simplest multi-layer perceptrons. In fact, as far as I remember, models couldn't generalize good enough without activation functions. SiLU is similar to ReLU, but when negative values approach 0, there are not zeroed, but get small negative value instead. 

```cpp
__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024)
    {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

// in-place, overwriting a
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}
```

---

[← GQA](18-gqa.md) · [🏠 Index](../README.md) · [Softmax →](20-softmax.md)
