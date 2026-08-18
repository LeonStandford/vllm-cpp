# 📚 Static batching

Instead of processing one request at a time, we process N requests. The pros: bigger throughput (more user requests processed at the same time). The cons: higher latency (all prompts in the batch need to wait until the longest prompt finishes processing, before being returned to the user). What you need is to basically in every place where we assumed there is a single token processed at a time, process multiple ones. Same goes for prefill - for multiple prompts, etc.

< TODO write more >

---

[← Buffer reuse](24-buffer-reuse.md) · [🏠 Index](../README.md) · [Continuous batching →](26-continuous-batching.md)
