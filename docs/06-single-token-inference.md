# 🎯 Single token inference

We start working on inference now. You can either close this course now and start coding, based on the operations sequence we laid out in [Safetensors and your model section](03-safetensors-and-your-model.md) or you can keep reading to fill your brain cache with useful stuff based on which you will have an easier time implementing the model. Your choice 

Regardless of what you choose, I'd like to share with you what is like to me to learn new things using LLMs as of 2026, maybe you'll find it useful when working on this course. I think about the learning process as of this loop:

1. Understand more-or-less what you want to build
2. Describe your mental model to a chatbot, ask it to figure out your blind spots, mistakes in your thinking, fill gaps in your knowledge with tailored information and enough context to understand
3. Read the answer, try to interalize it and update your mental model
4. Repeat until no mistakes
5. Start coding, continue until blocked
6. When blocked, go back to 2.

One thing that I find most important is to remember that a human learns by putting a real effort into understanding. Figuring things out, making mistakes, debugging our thought processes, thinking about our thinking, updating our understanding of things by describing them to others and recognizing the spots where our understanding is insufficient or wrong. All these things we need to do intentionally, because we can always delegate it to LLM. I am strong proponent of LLMs as a personalized tutor and it's a best thing to learn with, as of 2026. Also, I think there are 2 different types of effort and the confusion about whether and how to learn with LLMs arises. One type of effort is the effort of understanding, where you rewire your brain by thinking about a thing you want to understand. You break it down into smaller pieces, understand the mechanism, the data flow, the relationships, you relate it to other things you know, you imagine, you recreate, you make mistakes and they point you where you should look next. Digging deep into a concept you want to learn is one of the highest beauties of life. This type of effort and struggle is where learning happens. And then there's also a waste energy effort. Things like - data retrieval, commute, mechanistic actions that eat up your time, which you are able to fully do by yourself, you understand them from the inside out, and they learn you anything new anymore - this is a type of effort I think has too low ROI to perform, and it's better to delegate it to LLM. You can argue that in some sense it's still delegating your thinking to LLM and you're not wrong. You can always get better at reading documentation (you will do that anyway when learning, but often you can skip this part for one-off things) or internet search. But your time, focus and energy is limited. If becoming marginally better at internet search comes at the expense of learning more about the current topic you learn - then it's not a good tradeoff. YMMV though, not an expert, I'm just an internet random

You have a comfort of always being able to take a look at my code if you want to understand how I built it. Most of the time, you probably don't need to. But if you're stuck or when you finish a certain thing and want to compare - then ofc feel encouraged to do that. 

I'll lead you through a blueprint of a program - from an empty page to single token inference. You have now a lot of info you needed to start. 

So, you need to bootstrap the project. Import CUDA runtime, cuBLAS and `nlohmann::json`. This is the kind of work you can freely copy-paste from my code, if you don't want to do it.

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0 // this one is useful, but I don't remember why
#include "json.hpp"

using json = nlohmann::json;
```

If you're using VS Code and its complaining about unresolved imports, you need to adjust paths in `.vscode/` files. 

A useful helper function that you could run once at the beginning of your `main()` function is some debug info about the GPU your CUDA chose:

```cpp
int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}
```

Ok, now load the model. You know the safetensors file structure. There are many possible approaches to how you will store the model. The approach I like and I think you might like as well is to load the tensors as a single big block to the GPU memory. You need to allocate this memory first and safetensors file header will tell you how much. Then, using offsets in the header you will map the regions in GPU memory into pointers on CPU, so you know where to look when you want to, say, retrieve weights of K in layer 5. Some weights are not layer-specific, make sure to recognize that. One tip: make yourself a favor and print a lot. Debuggers are great, but not that much useful for working with raw data. Write (or generate by LLM) helpers scripts in Python to print dumps of models. I do lots of such things and they help me understand the data, model, etc. Anything that helps you move forward and strenghtens your understanding is ok, if you ask me. 

Oh btw, the data type of model is bfloat. In CUDA, it's `__nv_bfloat16`

If you implement weights in a similar way to what I described above, then to retrieve weights of K in layer, you'll write something alone these lines:

```cpp
weights.w_k[5] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers.5.self_attn.k_proj.weight"));
```

You need to allocate data for all tensors your model uses. Don't be afraid to overallocate and make it suboptimal. The first goal is to make it work.

In section [Safetensors and your model](03-safetensors-and-your-model.md) we laid out the exact sequence of operations we need to implement to predict the first token. Once we make it work and actually get our first token generated, we will reuse a lot of the code and write a loop around it.

---

[← GPU and CPU memory](05-gpu-and-cpu-memory.md) · [🏠 Index](../README.md) · [Tokenization →](07-tokenization.md)
