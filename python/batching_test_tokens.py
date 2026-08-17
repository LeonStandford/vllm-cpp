from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

prompts = [
    "What is 2+2?",
    "Name a color.",
    "Say hello.",
    "Capital of France?",
]

offset = 0
for i, p in enumerate(prompts):
    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    tokens = t.encode(text, add_special_tokens=False)
    print(f"// PROMPT {i} ({p}) - length {len(tokens)}")
    for tok in tokens:
        print(f"input_tokens.push_back({tok});")
    print(f"prompt_offsets.push_back({offset});")
    print(f"prompt_lengths.push_back({len(tokens)});")
    print()
    offset += len(tokens)


# // PROMPT 0 (What is 2+2?) - length 17
# input_tokens.push_back(128000);
# input_tokens.push_back(128006);
# input_tokens.push_back(882);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# input_tokens.push_back(3923);
# input_tokens.push_back(374);
# input_tokens.push_back(220);
# input_tokens.push_back(17);
# input_tokens.push_back(10);
# input_tokens.push_back(17);
# input_tokens.push_back(30);
# input_tokens.push_back(128009);
# input_tokens.push_back(128006);
# input_tokens.push_back(78191);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# prompt_offsets.push_back(0);
# prompt_lengths.push_back(17);

# // PROMPT 1 (Name a color.) - length 14
# input_tokens.push_back(128000);
# input_tokens.push_back(128006);
# input_tokens.push_back(882);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# input_tokens.push_back(678);
# input_tokens.push_back(264);
# input_tokens.push_back(1933);
# input_tokens.push_back(13);
# input_tokens.push_back(128009);
# input_tokens.push_back(128006);
# input_tokens.push_back(78191);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# prompt_offsets.push_back(17);
# prompt_lengths.push_back(14);

# // PROMPT 2 (Say hello.) - length 13
# input_tokens.push_back(128000);
# input_tokens.push_back(128006);
# input_tokens.push_back(882);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# input_tokens.push_back(46864);
# input_tokens.push_back(24748);
# input_tokens.push_back(13);
# input_tokens.push_back(128009);
# input_tokens.push_back(128006);
# input_tokens.push_back(78191);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# prompt_offsets.push_back(31);
# prompt_lengths.push_back(13);

# // PROMPT 3 (Capital of France?) - length 14
# input_tokens.push_back(128000);
# input_tokens.push_back(128006);
# input_tokens.push_back(882);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# input_tokens.push_back(64693);
# input_tokens.push_back(315);
# input_tokens.push_back(9822);
# input_tokens.push_back(30);
# input_tokens.push_back(128009);
# input_tokens.push_back(128006);
# input_tokens.push_back(78191);
# input_tokens.push_back(128007);
# input_tokens.push_back(271);
# prompt_offsets.push_back(44);
# prompt_lengths.push_back(14);