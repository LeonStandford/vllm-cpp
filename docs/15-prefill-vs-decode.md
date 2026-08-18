# ⏩ Prefill vs decode

An interesting fact about LLM inference is that it's not exactly the same process for the first predicted token vs all the next tokens. To predict the first token, you need to process all the input tokens. The input tokens are user's prompt or the chat history. All the computation that we need to do to get the first predicted token is called _prefill_. Everything that will happen after is called _decode_. 

Most of the computation, like Q projection, attention, attention scores, feed-forward (MLP) is thrown away as soon as it's passed to the next operation - both in prefill and decode. The useful mental model is that the only thing that you preserve at every stage of LLM inference is K projection, V projection and what is the last generated token. That's it. There are interesting implications of it, for instance - you could stop the inference, copy your K and V projections and last generated token, restart the server, load them into the server and use the last generated token as the input and you'd get the same next token predicted, as in the original server instance. I hope some of you challenge my claim and actually test it - let me know if you do :D

---

[← The column-major to row-major transposition trick](14-the-column-major-to-row-major-transposition-trick.md) · [🏠 Index](../README.md) · [Why KV cache exists →](16-why-kv-cache-exists.md)
