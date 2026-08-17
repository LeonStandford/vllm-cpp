"""
Reference outputs for verifying tiny-vllm kernel correctness.
Usage:
    python reference.py "The capital of France is" --model meta-llama/Llama-3.2-1B-Instruct
Prints embedding, RMSNorm, and Q/K/V projection values for comparison with C++ output.
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

        # Q projection: normed @ wq^T
        layer0 = model.model.layers[0]
        q = layer0.self_attn.q_proj(normed)
        print(f"\nQ projection shape: {q.shape}")
        for t in range(len(ids)):
            print(
                f"\nToken {t} (id={ids[t]}) Q projection first {args.num_values} values:"
            )
            for i in range(args.num_values):
                print(f"  [{i}] = {q[0, t, i].item()}")

        # K projection: normed @ wk^T
        k = layer0.self_attn.k_proj(normed)
        print(f"\nK projection shape: {k.shape}")
        for t in range(len(ids)):
            print(
                f"\nToken {t} (id={ids[t]}) K projection first {args.num_values} values:"
            )
            for i in range(args.num_values):
                print(f"  [{i}] = {k[0, t, i].item()}")

        # V projection: normed @ wv^T
        v = layer0.self_attn.v_proj(normed)
        print(f"\nV projection shape: {v.shape}")
        for t in range(len(ids)):
            print(
                f"\nToken {t} (id={ids[t]}) V projection first {args.num_values} values:"
            )
            for i in range(args.num_values):
                print(f"  [{i}] = {v[0, t, i].item()}")

        # Print weight shapes for reference
        print(f"\nWeight shapes:")
        print(f"  q_proj: {layer0.self_attn.q_proj.weight.shape}")
        print(f"  k_proj: {layer0.self_attn.k_proj.weight.shape}")
        print(f"  v_proj: {layer0.self_attn.v_proj.weight.shape}")
        print(f"  o_proj: {layer0.self_attn.o_proj.weight.shape}")
        print("-" * 50)
        # Layer-by-layer hidden state comparison
        # Create position IDs and get position embeddings
        hidden = embeds
        position_ids = torch.arange(len(ids)).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        for i, layer in enumerate(model.model.layers):
            print(f"\nLayer {i} input first 10 values (token 0):")
            for j in range(10):
                print(f"  [{j}] = {hidden[0, 0, j].item()}")
            hidden = layer(hidden, position_embeddings=position_embeddings)[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            print(f"  shape after layer: {hidden.shape}")

        # Final hidden state after all layers
        print(f"\nFinal hidden state first 10 values (token 0):")
        for j in range(10):
            print(f"  [{j}] = {hidden[0, 0, j].item()}")

        # Final RMSNorm
        final_normed = model.model.norm(hidden)
        print(f"\nFinal RMSNorm first 10 values (token 0):")
        for j in range(10):
            print(f"  [{j}] = {final_normed[0, 0, j].item()}")
        output = model(input_ids)
        logits = output.logits[0, -1, :]  # last token's logits
        predicted = torch.argmax(logits).item()
        print(f"Predicted token ID: {predicted}")
        print(f"Decoded: {tokenizer.decode([predicted])}")


if __name__ == "__main__":
    main()
