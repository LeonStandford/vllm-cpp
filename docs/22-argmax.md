# 🏆 Argmax

Pick the token that has a highest score. 

```cpp
cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * VOCAB_SIZE, cudaMemcpyDeviceToHost);
max_token = (float)embed_proj_cpu[0];
max_token_idx = 0;
for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
{
    if ((float)embed_proj_cpu[token_idx] > max_token)
    {
        max_token = embed_proj_cpu[token_idx];
        max_token_idx = token_idx;
    }
}
std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
```

---

[← Causal mask](21-causal-mask.md) · [🏠 Index](../README.md) · [Feed forward network →](23-feed-forward-network.md)
