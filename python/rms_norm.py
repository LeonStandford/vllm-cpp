"""
Reference RMSNorm output for verifying tiny-vllm kernel correctness.
Usage:
    python reference_rmsnorm.py "The capital of France is" --model meta-llama/Llama-3.2-1B
Prints embedding values and post-RMSNorm values for comparison with C++ output.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to process")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--num-values", type=int, default=10, help="Number of values to print per token"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()

    ids = tokenizer.encode(args.text)
    print(f"Token IDs: {ids}")
    print(f"Num tokens: {len(ids)}")

    input_ids = torch.tensor([ids])

    with torch.no_grad():
        # Embedding lookup
        embeds = model.model.embed_tokens(input_ids)
        print(f"\nEmbedding shape: {embeds.shape}")
        for t in range(len(ids)):
            print(
                f"\nToken {t} (id={ids[t]}) embedding first {args.num_values} values:"
            )
            for i in range(args.num_values):
                print(f"  [{i}] = {embeds[0, t, i].item()}")

        # First layer RMSNorm (input_layernorm of layer 0)
        normed = model.model.layers[0].input_layernorm(embeds)
        print(f"\nRMSNorm output shape: {normed.shape}")
        for t in range(len(ids)):
            print(f"\nToken {t} (id={ids[t]}) RMSNorm first {args.num_values} values:")
            for i in range(args.num_values):
                print(f"  [{i}] = {normed[0, t, i].item()}")

        # Also print RMSNorm weight for reference
        norm_weight = model.model.layers[0].input_layernorm.weight
        print(f"\nRMSNorm weight first {args.num_values} values:")
        for i in range(args.num_values):
            print(f"  [{i}] = {norm_weight[i].item()}")


if __name__ == "__main__":
    main()
