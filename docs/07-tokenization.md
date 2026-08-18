# 🔤 Tokenization

Ok, so I assume you loaded the model and mapped the weights of the model to some useful pointers. Now we need to read user's input, the prompt, and turn it from a text to something that model understands - tokens. All mainstream LLMs use tokens, not words or characters.

To turn text into a sequence of tokens, you need a tokenizer. We will use an existing tokenizer, which produces tokens that match the dictionary of Llama 3.2 1B. The reference is Hugging Face's own tokenizer - the one `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")` gives you

The engine doesn't need Python at runtime though: [src/tokenizer.cpp](../src/tokenizer.cpp) is a C++ port of the same thing, reading Llama's `tokenizer.json` directly (the tiktoken-style split regex, the ByteLevel byte↔unicode alphabet, and the BPE merges). So `tiny-vllm` takes prompts as plain text and prints answers as plain text:

```sh
./build/tiny-vllm                       # runs the built-in demo prompts
./build/tiny-vllm "What is 2+2?" "Say hello."
```

Every token id it produces was cross-checked against Hugging Face `tokenizers` on the same `tokenizer.json`: the built-in demo prompts, three whole source files, and 3000 random strings covering accented Latin, CJK, emoji and whitespace edge cases - all identical, and `decode(encode(text)) == text` byte for byte.

Going deep into tokenizers is out of the scope, what you really need to remember is that it takes a text and produces a sequence of tokens (ints), which represent your text but as a vector of ints. And LLM needs your text as this vector of ints.

> Building your own tokenizer is quite a fun thing. I wrote mine 3 years ago and feel free to use it as a reference, if you'd like to learn more about tokenizers: https://github.com/jmaczan/bpe-tokenizer. There's also a great resource from Andrej Karpathy where he builds a tokenizer, and it's very useful and educational: video https://www.youtube.com/watch?v=zduSFxRajkE, code https://github.com/karpathy/minbpe and this article https://github.com/karpathy/minbpe/blob/master/lecture.md

---

[← Single token inference](06-single-token-inference.md) · [🏠 Index](../README.md) · [Embeddings →](08-embeddings.md)
