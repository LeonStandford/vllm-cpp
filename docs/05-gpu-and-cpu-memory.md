# 🧠 GPU and CPU memory

This chapter might be a bit random, but I hope it's useful if you still begin with GPU programming. Data can live in host or device. 

Host is your PC with CPU. It has slow, big memory - DRAM - the one you buy and plugin into a motherboard. Recently very price'y, because of big tech AI craze. DRAM is separate from CPU. CPU has also it's own, small and fast on-chip memory - SRAM (I will wait until you stop laughing, if you're Slavic). 

Device is your GPU. It also has slow, big memory - HBM, VRAM - but it's soldered to the card and you can extend it easily, as you do with DRAM in your PC. It has small, fast memory - SRAM - which you can use if you defined `__shared__` variables.

GPU can't access your DRAM, so before running any computation on GPU, you need to copy data to it. The typical flow can be like this:

1. Create a variable on CPU
2. Write to variable on CPU
3. Compute the size of memory taken by the variable on CPU, or figure the max size some other way (sometimes you know the maximum number of elements in vector or alike)
4. Multiply the computed size by the size of variable type
5. Allocate the memory on GPU
6. Copy the variable from CPU to GPU
7. You can use this data in your GPU computations now

Ideally you want to allocate as little memory as possible, reuse this memory as much as possible and copy data as rarely as possible.

Let's have an example: I need to know what tokens are currently active during the inference. For this purpose, I need this data both on CPU and GPU.

1. I create a vector on CPU:
```cpp
std::vector<int> active_tokens;
```
2. I write to the vector some tokens, like
```cpp
active_tokens.push_back(token);
```
3. Now I need to figure out how many active_tokens there might ever be (max number), to allocate the GPU memory accordingly. In this case, I know that it might be up to BATCH_SIZE (say, max 2 elements always)
4. Tokens are ints, so the final size of memory I need to allocate is
```cpp
BATCH_SIZE * sizeof(int)
```
5. Declare and allocate the GPU memory
```cpp
int *gpu_active_tokens;
cudaMalloc(&gpu_active_tokens, BATCH_SIZE * sizeof(int));
```

`cudaMalloc` is the CUDA function you use for allocating GPU memory. You need a pointer on host - `int *gpu_active_tokens` - to be able to use this memory and pass it to your GPU kernels.

6. Copy data from CPU to GPU. We know the maximum amount of active tokens (BATCH_SIZE), but in the case if it's currently less active tokens processed, we need to compute amount of them: `num_active_slots * sizeof(int)`

```cpp
//               destination,               source,           size of data to copy,   direction of copying
cudaMemcpy(gpu_active_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
```

7. Now you can use this data when you invoke CUDA kernels, like
```cpp
embeddingGatherDecode(gpu_active_tokens, num_active_slots, hidden_state, weights.embed_tokens);
```

---

[← How floating-point numbers work and why we use bfloat16](04-how-floating-point-numbers-work-and-why-we-use-bfloat16.md) · [🏠 Index](../README.md) · [Single token inference →](06-single-token-inference.md)
