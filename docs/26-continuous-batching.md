# 🔁 Continuous batching

Solves the problem of needing to wait for a longest prompt in batch to be processed. It solves it by having slots in a batch. You fill prompts into slots. Once generation in a given slot is finished, the result from it is returned to the user. A prompt that awaits in a queue gets selected for the freshly emptied slot. The prompt goes through the prefill (all other elements in batch wait until it's finished). Once prefill is done, batching continues with all the elements of batch, including the new one.

< TODO write more >

---

[← Static batching](25-static-batching.md) · [🏠 Index](../README.md) · [Online softmax →](27-online-softmax.md)
