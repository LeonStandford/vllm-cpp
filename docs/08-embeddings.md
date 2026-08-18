# 🧬 Embeddings

Your text is translated into tokens and you feed the tokens into your LLM inference server. Tokens are more like indices, but they are not the data on which your LLM is going to work on. Large language models know how to map each token to a vector, where every token has the same vector length, but different vector values. These vectors are called embeddings. They embed the meaning of a token. Then you feed a list of tokens into the LLM, it retrieves one embedding per token, where tokens work as indexes that tell the model which embedding (vector) it should retrieve from it's weights. In our case, every embeddings has 2048 length. So for 5 input tokens, you get 5 vectors of 2048 length, which together is a matrix of dimension (5, 2048). We already know the type of every number in these embedding vectors - its bfloat16.

This might be a first CUDA kernel to write in this course. Your job is to retrieve the embeddings for all the input tokens.

What do you need to do that? You need input tokens and embeddings weights. You already loaded embeddings weights on your GPU. But input tokens you provide live on CPU - you can provide them as CLI params, hardcode or read from a file. By default, you load them into some vector of ints, probably. Now you need to make them available to your GPU. So what you do now?

You can create a buffer on GPU where you copy your input tokens. When you do it, you will be able to use the pointer to these input tokens on GPU and pass the pointer to your CUDA kernels. This time, I will help you do it. Any other kernels and CUDA data move/allocation you will write on your own, unless they will be exceptionally interesting, okay?

Let's say you have your input tokens in a vector of ints on CPU:

```cpp
std::vector<int> input_tokens = {678, 264, 1933, 13};
```

Your model, like all models, has a constraint about how many tokens it can process. For Llama 3.2 1B it's 2048. It includes both input and output tokens. We need to arbitrarily choose how many of them we allow to be the prompt size. Let's say it's going to be max 512 tokens as an input and we declare it as a [constexpr](https://en.cppreference.com/w/cpp/language/constexpr.html):

```cpp
constexpr int MAX_PROMPT_LEN = 512;
```

You need a copy of your input tokens on GPU. So, you need to allocate a memory on GPU and you need a pointer to this memory. A pointer:

```cpp
int *gpu_input_tokens;
```

To allocate the memory on GPU, we will use `cudaMalloc` function, which you already know from a chapter about GPU memory. The first argument of `cudaMalloc` is a pointer to our pointer (`void **`). The second argument is a size of memory to allocate. We know how many tokens we can maximum have as an input. The tokens are integers, so the size of memory to allocate is max number of input tokens * size of an int.

```cpp
cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
```

Ready to copy input tokens into GPU now:

```cpp
//              destination,              source,                              size, a direction of copying
cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
```

We can write a CUDA kernel now. 

---

[← Tokenization](07-tokenization.md) · [🏠 Index](../README.md) · [CUDA kernel engineering - embeddings →](09-cuda-kernel-engineering-embeddings.md)
