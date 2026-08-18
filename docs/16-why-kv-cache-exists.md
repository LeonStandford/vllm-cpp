# 🗄️ Why KV cache exists

We can reuse some parts of the computation results to predict the next tokens. You don't have to reuse the results, but they don't change so computing them again and again is a pure waste. You already know that the only data that gets moved forward in the computation is K and V projection and last generated token. If we generate 1 token at the same time, both K and V are vectors of bfloat16s and last generated token is a single int. If we generate more tokens at the same time - in other words, if we do batching - both K and V are matrices of bfloat16s and last generated tokens is a vector of ints.

When we process a token, regardless of whether it's prefill or decode, from perspective of data we preserve (K, V projections and last generated token) it looks the same:

0. ...
1. Compute K projection using last generated token
2. Store it
3. Compute V projection using last generated token
4. Store it
5. ...
6. Use all K projections and all V projections to compute attention
7. ...
8. Generate new token
9. Store it as last generated token

Let's say we don't store K and V projection for current token. It would mean that we need to compute all K and V projections for the current and all previous tokens before we can compute attention for current token. Again, pure waste. That's the reason why store the K and V projections. It's just a record of all previous K and V projections. You don't modify it during the LLM inference. You just append to it, with every processed token. The name of this K and V projections storage is KV cache.

---

[← Prefill vs decode](15-prefill-vs-decode.md) · [🏠 Index](../README.md) · [Attention →](17-attention.md)
