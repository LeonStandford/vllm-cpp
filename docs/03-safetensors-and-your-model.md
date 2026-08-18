# 📦 Safetensors and your model

First thing you need to do is to download a LLM which we will use to run inference on. I choose Llama 3.2 1B Instruct, because it's easy, small, tuned for dialogs and good enough for us. From perspective of us, the engineers who build an inference server, the model is just a single file containing weights.

The model is in [Safetensors format](https://huggingface.co/docs/safetensors/index). There exist other formats, like [Pickle](https://docs.python.org/3/library/pickle.html) and [Parquet](https://parquet.apache.org/docs/file-format/). Safetensors is just very popular and widely used, and the model we picked is hosted in Safetensors

Let's stop for a moment and understand the Safetensors format before we move on.

A safetensor file consists of 3 sections, always in this order: header size, header and tensors data. Header size is always 8 bytes. These 8 bytes are an unsigned 64-bit integer, which says how many bytes the actual header takes.

```cpp
std::ifstream safetensors_file("model.safetensors", std::ios_base::binary);
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
```

The header is a JSON that contains about all the tensors inside the file. JSON is just a group of pairs <key, value>, where key is a unique string with a tensor name and value is another JSON object, containing info about this tensor. Every key in this JSON is a name of the tensor, except a single key which is called `__metadata__`, probably for some additonal info when necessary (we won't use it, specs say it's a "special key for storing free form text-to-text map). Every value is a JSON containing three keys - `dtype`, `shape` and `offsets`. `dtype` says what data type the tensor is stored in. `shape` says the [dimensions of a tensor](https://en.wikipedia.org/wiki/Tensor#As_multidimensional_arrays) and `offsets` say where the tensor is stored, within the tensors data section. Every `shape` is a list of ints of unknown length and every `offsets` value is a vector of exactly two ints. First element says where the tensor begins and last element says where the tensor ends.

You face the first design decision now. Do you want to make your inference server architecture independent, so it can run any arbitrary model, as long as you implement the operations it needs, or do you want to start simple and focus on our model of choice?

Whatever you decide, it's always easier to develop on a single model and then generalize, than try to make it flexible from the beginning when you're not sure how the code will look like at the end. You can always get back to it and update when you choose to.

If you want to make your server model-independent, you need to allocate the memory, set the model data `shape` and type (`dtype`) dynamically based on Safetensors header and implement more operations, making sure to cover all operations used by all models that you want to support. Probably you would still need to provide some blueprints of the models architecture, because the Safetensors file doesn't tell you which operation to run with this data, in what order etc. I'm not sure what's the optimal approach to do that, but if you figure it out, either on your own or by reading through vLLM/[TensorRT](https://github.com/NVIDIA/TensorRT)/others code, feel invited to share your findings

I will assume that we just code our server for Llama 3.2 1B Instruct architecture. Here's a dump of [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) `LlamaForCausalLM` object with `meta-llama/Llama-3.2-1B-Instruct` model loaded. We can inspect which operations we need to implement and what data shape and data types we need to use:

```py
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
```

Let's dissect it.

First of all, we don't really know neither the order of operations or data type from this. But! [Model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face page tell us that weights are in [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) format. We will go back to the format soon. 

We need to understand the order of operations to know how to code it. Sebastian Raschka has a gallery of LLM architectures and it shows nicely how the operations are organized - see [here (the left one)](https://magazine.sebastianraschka.com/i/168650848/61-qwen3-dense).

By looking at the diagram from Sebastian, we see that the operations order in LLama 3.2 1B is like this:

1. Send some text to the model
2. Turn it into tokens (a new concept, we didn't mention it yet)
3. Retrieve an embedding for each token
4. 16 transformer blocks, also called layers, which consist of:
  - RMS Norm
  - Residual connection
  - Masked grouped-query attention, which consists of:
    - Q projection
    - K projection
    - V projection
    - RoPE with Q projection
    - RoPE with K projection
    - Attention
    - Attention scores
    - Causal mask
    - Softmax
    - Residual connection
    - Attention scores with V projection
  - O projection (output projection)
  - Residual connection add
  - RMS Norm
  - [Feed forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) (like in first neural networks, [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)), which consists of:
    - Gate projection, first linear layer
    - Up projection, second linear layer
    - [SiLU](https://arxiv.org/pdf/1702.03118) activation function, similar to [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) but it's looking more like a sigmoid
    - Down projection, third linear layer
    - Residual connection add
5. RMS Norm
6. Linear output
7. Argmax

After these steps, we should get our first token produced by the language model ran on our server.

Don't worry if some of this operations are not familiar yet. You will understand them viscerally once we progress through the course. I keep forgetting how they work, too, and often need to look up again, so don't feel bad for going back and forth or looking things up with a chatbot or internet search.

---

[← Technical prerequisities](02-technical-prerequisities.md) · [🏠 Index](../README.md) · [How floating-point numbers work and why we use bfloat16 →](04-how-floating-point-numbers-work-and-why-we-use-bfloat16.md)
