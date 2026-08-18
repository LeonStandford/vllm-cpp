# 📐 RMSNorm and parallel reduction in CUDA

Look back at the sequence of operations in our model (section [Safetensors and your model](03-safetensors-and-your-model.md)). After we retrieve the embeddings for our tokens, it's time for [RMSNorm](https://arxiv.org/abs/1910.07467). Unlike embeddings gather, it's a first operation that will run in layers. Our model, Llama 3.2 1B, has 16 layers. RMSNorm takes our retrieved embeddings and - using model weights for a rms norm `weights.input_layernorm[layer]` - runs RMSNorm function. RMSNorm is an operation that modifies all numbers in an embedding. To do that, first it needs to see all the elements and compute their [root mean square](https://en.wikipedia.org/wiki/Root_mean_square) sum.

Based on the paper, the formula is:

$$ \text{normalized}_i = \frac{a_i}{\text{RMS(a)}} \text{, where RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i} $$

Let's write this kernel together. 

To compute RMSNorm, we need to compute `RMS(a)` first. It requires going through all the numbers, so we will need some synchronization between threads. We normalize each embedding separately. So again, we launch as many blocks as there are input tokens. Embeddings have 2048 numbers, but 1024 is max threads per block, so we will use the same trick as we used in embeddings gather kernel - we will process two numbers within each thread. We need some temporary value, to which we can write the squares of every number of an embedding. We could use a single variable for it. Every thread within a block would write a square of its number to it. The problem here is synchronization of reads and writes to the shared variable - preventing race condition, ensuring that we will end up with a correct sum. There is another approach, where we create a temporary vector of the same length as number of threads in a block, and each thread writes a square of its value to `threadIdx.x` position in a temporary vector. This way, we ensure that no two threads will try to read or write to the same position in the vector. This technique is called [parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (tree reduction).

Let's create a vector. We use `__shared__` keyword to indicate that the vector is shared among all threads of a block.

One thing about numerical stability before we move on - remember that our data is bfloat16 format. Main strengths of bfloat16 is a big exponent - 8 bits, same size as float (float32) - but the mantissa is small (7 bits only vs 10 bits in float16). It means that to not lose a precision in our computation, we can use a bigger type to actually compute things inside the kernel - things like a square of each number. We'll use float for this purpose. We will cast every number to float and declare our temporary buffer as float, too.

```cpp
__shared__ float rms_vector[1024];
rms_vector[threadIdx.x] = (float)input[threadIdx.x] * (float)input[threadIdx.x];
```

We wanted to compute two numbers per thread, so we either make `rms_vector` bigger (2048 items) or append the square of a number 1024 indices away to our `rms_vector[threadIdx.x]`. The second one is easier to implement and will be probably much faster, because we will have less steps in a tree reduction - it's faster to reduce to 1 element from 1024 elements than from 2048 elements.

```cpp
__shared__ float rms_vector[1024];
rms_vector[threadIdx.x] = (float)input[threadIdx.x] * (float)input[threadIdx.x] + (float)input[threadIdx.x + 1024] * (float)input[threadIdx.x + 1024];
```

It would work, if we launched just a single block. But again, we want to be able to process multiple tokens, so we will launch as many blocks as tokens. Because of that, we need to figure out the correct item indexes for current thread. `threadIdx.x` is correct as a index of `rms_vector`, because `rms_vector` is always 1024 floats. `threadIdx.x` won't do as an input index, because we have multiple tokens on the input. We know each tokens takes 2048 `__nv_bfloat16`s of space. An index of a block tells us an index of a token. So, to move to the current token in the input, we need to multiply `blockIdx.x` by size of a token - by 2048. Our input index for a current thread is then:

```cpp
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

Now, the reduction part. The idea is that every i-th element adds an element at the index of `self + i` to itself. And then, we multiply i by 2 and repeat. After every iteration, we need to make sure that all the threads finish writing to the `rms_vector` before we move to the next iteration. CUDA has [`__syncthreads()`](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#thread_synchronization), which we can use for this purpose, and we put it after we write to `rms_vector`.. It can be done in a loop, but we will start with writing out everything explicitly, so the tree reduction algo will feel more understandable. The algorithm finished when we get to the `self + 1024`. The sum of all elements is stored at 0 index - `rms_vector[0]`. Let's write down the code so you can actually see it yourself. To me, many pieces weren't obvious until I wrote it down myself. Maybe you'd like to try it to, before reading the code? 

Please do whatever you feel more comfortable with, and I will write down the verbose (but correct) version of the RMSNorm kernel anyway:

```cpp
__shared__ float rms_vector[1024];
int workIndex = threadIdx.x + blockIdx.x * 2048;
rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
__syncthreads();
```

Each thread in a block stores its input number's square summed with a square of the input number, shifted right by 1024 (covering the second part of 2048 vector, which we can't cover by launching 2048 threads in a block, because of CUDA limitation to max 1024 threads per block). `rms_vector[0]` contains 2 squares out of 2048 numbers of a currently processed input embedding (for a current token) - indices `0` and `1024`.

We can go to the first step of reduction. Every second thread will add the next number in the `rms_vector`. Remember that every number in rms_vector currently already stores two squares.

```cpp
if (threadIdx.x % 2 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1]; 
}
__syncthreads();
```

`%` is modulo operator ([division with remainder](https://en.wikipedia.org/wiki/Euclidean_division)). Now `rms_vector[0]` stores squares of numbers on indices `0`, `1024` and `1`. We know that `1` already contained `1` and `1025`, so `rms_vector[0]` has actually these 4 squares now: `0`, `1`, `1024` and `1025`.

We increase the "gap" by 2, so now every 4-th element will add an element that is 2 indices away from self:

```cpp
if (threadIdx.x % 4 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2]; 
}
__syncthreads();
```

Notice that the index we add to the `threadIdx.x` is half of what we divide with (we divide with reminder by 4 and add the index that is 2 items away). Think about why do we do this and if the summation would give the correct result when we would add the same number by which we modulo.

The process continues until we modulo `threadIdx.x` by 1024:

```cpp
if (threadIdx.x % 2 == 0)
{
    // every second item has its predecessor
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1];
}
__syncthreads();
if (threadIdx.x % 4 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2];
}
__syncthreads();
if (threadIdx.x % 8 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 4];
}
__syncthreads();
if (threadIdx.x % 16 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 8];
}
__syncthreads();
if (threadIdx.x % 32 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 16];
}
__syncthreads();
if (threadIdx.x % 64 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 32];
}
__syncthreads();
if (threadIdx.x % 128 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 64];
}
__syncthreads();
if (threadIdx.x % 256 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 128];
}
__syncthreads();
if (threadIdx.x % 512 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 256];
}
__syncthreads();
if (threadIdx.x % 1024 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 512];
}
__syncthreads();
```

Now `rms_vector[0]` holds the sum of all the squares of its token's embedding numbers. To compute $\text{RMS(a)}$ we need to divide it by a size of an embedding (2048) and take a square root of it: $\text{RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i}$.

```cpp
if (threadIdx.x == 0)
{
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0); // = RMS(a)
}
__syncthreads();
```

Now all threads can read `rms_vector[0]` to retrieve $\text{RMS(a)}$. Let's finish our kernel with computing a normalized value for current thread's numbers (the one that corresponds to threadIdx.x and threadIdx.x + 1024). Remember that we want to use `float` when doing math that might be affected by limited precision of `bfloat16`. So, we cast all numbers to `float`s and, before writing to the output memory, we cast it back to `__nv_bfloat16`. 

```cpp
output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);

output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
```

Almost done!

There is a problem with these two lines. Let's recall at how the $\text{RMS(a)}$ is used when computing a normalized value: $\text{normalized}_i = \frac{a_i}{\text{RMS(a)}}$. If $\text{RMS(a)}$ is 0, then this division becomes a division by 0 and in C++ we'll get NaN or infinity, both means that we lost and even one NaN will corrupt the inference and our engine will produce a garbage or crash. To prevent that from happening, the refernce LLM we use has an epsilon parameter, which defines what value we add to the result of RMSNorm. 

```py
(input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
```

This way, we prevent division by zero and NaNs propagation from happening. Let's just update a line where we compute $\text{RMS(a)}:

```cpp
if (threadIdx.x == 0)
{
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5); // = RMS(a)
}
__syncthreads();
```

Congrats on finishing your RMSNorm CUDA kernel! For a verbosity, we can replace the manual writing down every iteration of the reduction with a for loop and we're done here

```cpp
__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
    __syncthreads();
    // tree reduction
    for (int i = 1; i < 1024; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0)
        {
            rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
    }
    __syncthreads();

    output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
}
```

---

[← CUDA kernel engineering - embeddings](09-cuda-kernel-engineering-embeddings.md) · [🏠 Index](../README.md) · [RoPE →](11-rope.md)
