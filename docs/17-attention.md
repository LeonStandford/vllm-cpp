# 👀 Attention

Attention is an important part of LLM inference. It's where you do a lot of matrix multiplication using Q, K and V projections you computed earlier. A basic formula for scaled dot-product attention that comes from a paper [Attention is all you need](https://arxiv.org/pdf/1706.03762) is:

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

< TODO: this section is difficult to write, but I want to make it good and useful to you. Putting code here now, and want to come back to writing it Later™ >


```cpp
for (int i = 0; i < NUM_Q_HEADS; ++i)
{
    int k_head_idx = i / GQA_Q_TO_K_RATIO; // i / 4 <- it means 4 Q heads uses the same 1 K head
    __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
    __nv_bfloat16 *k_head = k_proj[layer] + k_head_idx * HEAD_DIM;
    __nv_bfloat16 *attn_score_head = attn_scores + input_tokens.size() * input_tokens.size() * i;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    input_tokens.size(),
                                                    input_tokens.size(),
                                                    HEAD_DIM,
                                                    &attn_alpha,
                                                    k_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    q_head,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &attn_beta,
                                                    attn_score_head,
                                                    CUDA_R_16BF,
                                                    input_tokens.size(),
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
}

// causal mask, softmax and then attention scores * V
        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 Q heads share 1 V head
        // attn_scores dim (32, num_tok, num_tok)
        // attn_scores head dim (num_tok, num_tok)
        // V dim (num_tok, 512)
        // NUM_V_HEADS is 8 -> 512 / 8 = 64
        // V_head dim (num_tok, 64)
        // output head dim: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in total 32 output heads: so (num_tok, 64 * 32) = (num_tok, 2048)
for (int i = 0; i < NUM_Q_HEADS; ++i)
{
    int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO; // GQA, 4 Q heads for 1 V head
    // i * input_tokens.size() * input_tokens.size(),  because attn scores is (32, num_tok, num_tok)
    __nv_bfloat16 *attn_scores_head = attn_scores + i * input_tokens.size() * input_tokens.size();
    __nv_bfloat16 *v_head = v_proj[layer] + v_head_idx * HEAD_DIM;
    __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    HEAD_DIM,
                                                    input_tokens.size(),
                                                    input_tokens.size(),
                                                    &attn_scores_v_alpha,
                                                    v_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    attn_scores_head,
                                                    CUDA_R_16BF,
                                                    input_tokens.size(),
                                                    &attn_scores_v_beta,
                                                    output_attn_scores_head,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
}
```

---

[← Why KV cache exists](16-why-kv-cache-exists.md) · [🏠 Index](../README.md) · [GQA →](18-gqa.md)
