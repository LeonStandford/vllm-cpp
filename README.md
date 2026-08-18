# 🚀 vllm-cpp

> A high performance LLM inference engine written from scratch in C++ and CUDA - a younger and smaller sibling of [vLLM](https://github.com/vllm-project/vllm)

This repository is two things at once:

1. 🛠️ **an inference engine** - a real, working Llama 3.2 1B server
2. 📚 **a course** - 30 chapters walking you through building it, deriving every idea and every bit of maths from scratch

Use it as a learning path, or as a teaching resource at your university. Make yourself a hot beverage and let's begin.

---

## ✨ What's inside

| | Feature |
|---|---|
| ✅ | Load a real LLM from Safetensors (Llama 3.2 1B Instruct) |
| ✅ | Full forward pass - prefill + decode |
| ✅ | Every operation as a hand-written CUDA kernel |
| ✅ | Byte-level BPE tokenizer in C++ - text in, text out |
| ✅ | KV cache |
| ✅ | Static batching |
| ✅ | Continuous batching |
| ✅ | [Online softmax, FlashAttention-like](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) |
| ✅ | [PagedAttention](https://arxiv.org/pdf/2309.06180) |

---

## ⚡ Quick start

```sh
./scripts/download_model.sh                 # 🤗 needs HF_TOKEN in .env (gated repo)
./scripts/build.sh                          # 🔨 cmake + ninja
./scripts/run.sh                            # ▶️  built-in demo prompts
./scripts/run.sh "What is 2+2?"             # 💬 your own prompt
```

```
Prompt (17 tokens): What is 2+2?

=== slot 0 ===
Prompt: What is 2+2?
Answer: 2+2=4
```

Every script works from any directory. The engine reads its weights from `.cache/huggingface/download/`.

<details>
<summary>🧪 Other scripts</summary>

| Script | What it does |
|---|---|
| `scripts/test.sh` | build, then run |
| `scripts/full_test.sh` | build, then run one fixed prompt |
| `scripts/check.sh` | run under `compute-sanitizer` |
| `scripts/nsys.sh` | profile with Nsight Systems |
| `scripts/ncu.sh` | profile kernels with Nsight Compute |

</details>

---

## 📁 Layout

```
src/main.cpp        engine - weight loading, prefill, decode, batching, paged KV cache
src/kernels.cu      CUDA kernels - embeddings, RMSNorm, RoPE, SiLU, softmax, causal mask
src/tokenizer.cpp   byte-level BPE tokenizer reading Llama's tokenizer.json
include/            json.hpp, tokenizer.hpp, generated unicode tables
scripts/            build / run / download / profiling
docs/               📚 the course
```

---

## 📚 The course

**Foundations**

| | Chapter | |
|---|---|---|
| 🧭 | [Intro: LLM, vLLM, models, inference servers](docs/01-intro-llm-vllm-models-inference-servers.md) | what we are building and why |
| 🧰 | [Technical prerequisities](docs/02-technical-prerequisities.md) | what you need to know first |
| 📦 | [Safetensors and your model](docs/03-safetensors-and-your-model.md) | reading real model weights |
| 🔢 | [How floating-point numbers work and why we use bfloat16](docs/04-how-floating-point-numbers-work-and-why-we-use-bfloat16.md) | bits, exponents, precision |
| 🧠 | [GPU and CPU memory](docs/05-gpu-and-cpu-memory.md) | where the data lives |
| 🎯 | [Single token inference](docs/06-single-token-inference.md) | the whole picture, end to end |

**From text to tensors**

| | Chapter | |
|---|---|---|
| 🔤 | [Tokenization](docs/07-tokenization.md) | text → token ids |
| 🧬 | [Embeddings](docs/08-embeddings.md) | token ids → vectors |
| ⚙️ | [CUDA kernel engineering - embeddings](docs/09-cuda-kernel-engineering-embeddings.md) | your first real kernel |
| 📐 | [RMSNorm and parallel reduction in CUDA](docs/10-rmsnorm-and-parallel-reduction-in-cuda.md) | reductions done right |
| 🌀 | [RoPE](docs/11-rope.md) | rotary position embeddings |
| ➕ | [Residual connections](docs/12-residual-connections.md) | the residual stream |

**Matrix multiplication**

| | Chapter | |
|---|---|---|
| ✖️ | [cublasGemmEx](docs/13-cublasgemmex.md) | letting cuBLAS do the heavy lifting |
| 🔄 | [The column-major to row-major transposition trick](docs/14-the-column-major-to-row-major-transposition-trick.md) | the trick that makes it all fit |

**Attention**

| | Chapter | |
|---|---|---|
| ⏩ | [Prefill vs decode](docs/15-prefill-vs-decode.md) | two very different phases |
| 🗄️ | [Why KV cache exists](docs/16-why-kv-cache-exists.md) | the single biggest win |
| 👀 | [Attention](docs/17-attention.md) | Q, K, V and scores |
| 👥 | [GQA](docs/18-gqa.md) | grouped query attention |

**The rest of the block**

| | Chapter | |
|---|---|---|
| 〰️ | [SiLU](docs/19-silu.md) | the activation function |
| 🌡️ | [Softmax](docs/20-softmax.md) | turning scores into weights |
| 🎭 | [Causal mask](docs/21-causal-mask.md) | no peeking at the future |
| 🏆 | [Argmax](docs/22-argmax.md) | picking the next token |
| 🕸️ | [Feed forward network](docs/23-feed-forward-network.md) | SwiGLU MLP |

**Making it a server**

| | Chapter | |
|---|---|---|
| ♻️ | [Buffer reuse](docs/24-buffer-reuse.md) | stop allocating in the hot loop |
| 📚 | [Static batching](docs/25-static-batching.md) | many prompts at once |
| 🔁 | [Continuous batching](docs/26-continuous-batching.md) | iteration-level scheduling |
| 🌊 | [Online softmax](docs/27-online-softmax.md) | one pass, no huge matrix |
| 📄 | [Paged Attention](docs/28-paged-attention.md) | virtual memory for KV cache |
| 🧱 | [Paged KV cache](docs/29-paged-kv-cache.md) | blocks and block tables |
| 🚀 | [Paged Attention CUDA kernel](docs/30-paged-attention-cuda-kernel.md) | putting it together |

---

## 📄 License

See [LICENSE](LICENSE).
