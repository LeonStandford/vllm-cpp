# ⚙️ CUDA kernel engineering - embeddings

> There exist much better resources than I can produce, so to learn CUDA, please check out [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html). I'll just briefly mention a basics here, but they might not be sufficient for you

Kernels are functions that are executed on a GPU. You launch the same function multiple times. Every launched function is a separate thread. They run the same code. They receive slightly different parameters, like index of a thread. Threads are grouped into blocks. When you launch a CUDA kernel, you define how many blocks you want to invoke and how many threads are there in every block. Threads are also grouped into warps. Every warp has 32 threads. So, when you run your kernel and you define that it has to run 5 blocks and each block has to run 64 threads, then it means that each block runs 2 warps, 32 threads in each warp. 

When writing CUDA kernels, a lot of effort goes into thinking about memory. I mean, it's like thinking from the thread perspective: "what data should I process?", "where I should write the results?". It in practice means figuring out index of input data you need to read and where you should write the output to. Your main tools are built-in variables, like threadIdx, blockIdx and blockDim - see this link for more structured info https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html. Every thread has it's own values of these variables. This makes possible to running the same computation in parallel. This approach is called [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads).

Ok, so what we know is that for every input token, we want to retrieve an embedding which consists of 2048 bfloat16 numbers. The first approach that we can think about is - okay, so we have N tokens to retrieve, and each token needs to retrieve 2048 numbers. So we can run N blocks - one block per token - and 2048 threads in every block, so every thread would retrieve exactly a single number. Let's write an empty kernel and write down how we would like to execute it with N blocks and 2048 threads per block.

Empty kernel:
```cpp
__global__ void embeddingGatherKernel()
{
}
```

An invocation of this kernel:
```cpp
embeddingGatherKernel<<<num_input_tokens, 2048>>>();
```

Looks good. Now let's think what data we need on the input and what we want to produce. We need input tokens and weights of embeddings of the loaded model. We also need some output to write to. We already copied tokens to GPU and we have weights of embeddings on GPU, too. So the output, what it should be like? We retrieve 2048 bfloat16 numbers for N tokens. So we need to have a memory, to which we can write that many numbers. Let's allocate it first:

```cpp
__nv_bfloat16* input_embeddings;
cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * 2048);
```

Size of memory we allocate: max number of input tokens (MAX_PROMPT_LEN) * 2048 * size of every number (sizeof(__nv_bfloat16)).

So let's pass these pointers into our kernel. `weights.embed_tokens` are weights of embeddings we have loaded from the safetensors file:

```cpp
embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, input_embeddings, weights.embed_tokens);
```

Let's update the kernel signature to accepth these arguments:
```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
```

Okay. So now we need to actually retrieve the numbers. Remember, we think about the kernel from the perspective of a single thread, where there are multiple threads invoked with exactly the same code, but their values of `threadIdx.x`, `blockIdx.x` (and a few similar variables) are different - that's how they know which thread they are and what data they need to take and what data they need to produce. 

Let's compute the index of output first. We know it's of dimension (number of tokens, 2048). Let's assume it's row-major format, so it's easier to work with it. In other words, it means when you access index 0, you get 0th token of 0th number, index 1, you get 1st token of 0th number, and like this up to 2047 index. Then, it starts again with 0th token of 1st number. `threadIdx.x` is unique within a block. So, if we had only a single token, we could map 1:1 `threadIdx.x` to `input_embeddings` index (the output memory), like input_embeddings[threadIdx.x]. For a moment, let's assume this is the case - that we always get a single token. So, we now know where we write to. We need to retrieve the correct number of an embedding for this token. Every embedding is 2048 numbers, so to get to the beginning of the embedding of id of the token, we need to multiply the token by 2048 - `gpu_input_tokens[0] * 2048` (again, assuming `gpu_input_tokens` is a single token, so that's why we can access the first element of it). This way we receive the index of the first number of the token's embedding. The embedding has 2048 numbers and each thread in a block retrieves a single number of this embedding. So, we need to move the index by the current value of `threadIdx.x`, which will tell us which number of the embedding we need to retrieve. So, the index becomes `gpu_input_tokens[0] * 2048 + threadIdx.x`. Let's combine it into a kernel implementation:

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    input_embeddings[threadIdx.x] = embed_tokens[gpu_input_tokens[0] * 2048 + threadIdx.x];
}
```

Looks good. Let's make it work for more than a input single token. What changes? We wanted to run multiple blocks - one block per input token. So we need to take this into account in index arithmetic. How will current thread's output index look like? Previously, it was just `threadIdx.x`. Now we need to take into account that there are multiple tokens. Every token results with 2048 sized embedding. So, we can multiply block's index by 2048 and add it to `threadIdx.x` to get a position in an embedding for currently processed token (and token idx == block idx). So it becomes:

```cpp
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

We can't hardcode first token anymore `gpu_input_tokens[0]`. Since token idx == block idx, we can replace `[0]` with `blockIdx.x`:

```cpp
embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
```

Combining it together and we get:

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
}
```

We create a function that is a wrapper over this kernel invocation, so you can run it from your C++ code. Kernels are in `.cu` files, because of their specific syntax, like the `<<<>>>` where you define the grid (blocks and threads). 

```cpp
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens)
{
    embeddingGatherKernel<<<???, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Wait a sec. We don't know a number of tokens. We don't use C++ data structures anymore, like vectors, so we can't read the number of elements of `gpu_input_tokens`. We need to pass it to this function explicitly:

```cpp
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Everything looks good now! Let's launch it. I encourage you to actually do it.

And then?

...

[It's a trap!](https://www.youtube.com/watch?v=4F4qzPbcFiA) Max threads per block is 1024, at least for most of NVIDIA GPUs you might have at home. In `checkGPUStatus()` function we event print this info: 

```cpp
std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
```

So, what we can do now? We can either launch 2 times more blocks or process 2 numbers in every thread instead of just one number. I go with the second option - it will be probably faster, doesn't require launching more threads and doesn't require any synchronization between threads or even between writing the output memory.

We have max 1024 threads per block. Embedding has 2048 numbers. We want to process two numbers from embedding per thread. What options do we have? We can either process current thread's number and the number next to it or we can process current thread's number and the number on the same position in the next half of the embedding, so current thread's index + 1024. Would the first approach work? First thread would process 0th and 1st number. Second thread would process 2nd and 3rd. Third 4th and 5th. Probably would work too (?), but more index arithmentic. The second option is easier again. Just add 1024 to both input and output indexes.

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
    input_embeddings[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}
```

Run it. This time it will work.

Congrats to you, you finished your first CUDA kernel!

Let's move back from computation and low-level programming to semantics/meaning of what we just did. Notice that while all these embeddings we retrieve for input tokens have some encoded meaning within them, they don't know about each other. They don't know their position within a text we provided as an input. They don't know what tokens they are surrounded with. They don't know the conversation history, etc etc. This understanding will be build and stored as K and V projections.

---

[← Embeddings](08-embeddings.md) · [🏠 Index](../README.md) · [RMSNorm and parallel reduction in CUDA →](10-rmsnorm-and-parallel-reduction-in-cuda.md)
