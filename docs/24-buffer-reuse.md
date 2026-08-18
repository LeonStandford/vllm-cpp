# ♻️ Buffer reuse

Sometimes, a buffer you allocate has the same size as a buffer that will be used later in the code. And sometimes, when second buffer starts to be needed after when data in the first buffer is not needed anymore (it was already used in a computation and won't be used anywhere else), then we can use the first buffer to write data, which we would write to the second buffer. This way, we can allocate less memory. It means we reuse the same buffer between two different places. We can safely do it, once we confirm by lifetime analysis that lifetimes of these two buffers don't overlap. See how it works in for `buf_2048_1` and `buf_2028_2` in [`src/main.cpp`](../src/main.cpp)

< TODO write more and explain lifetimes analysis >

---

[← Feed forward network](23-feed-forward-network.md) · [🏠 Index](../README.md) · [Static batching →](25-static-batching.md)
