"""
Tokenize text using HuggingFace tokenizers and dump token IDs.

Usage:
    python python/tokenizer.py "The capital of France is" --model meta-llama/Llama-3.2-1B
    python python/tokenizer.py --decode --ids 791 6864 315 9822 374 --model meta-llama/Llama-3.2-1B

The C++ engine reads/writes plain text files of token IDs.
"""

import argparse
import json
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--decode", action="store_true", help="Decode mode: IDs → text")
    parser.add_argument("--ids", nargs="+", type=int, help="Token IDs to decode")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.decode:
        if not args.ids:
            parser.error("--decode requires --ids")
        text = tokenizer.decode(args.ids)
        print(text)
    else:
        if not args.text:
            parser.error("Provide text to tokenize")
        ids = tokenizer.encode(args.text)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(ids, f)
            print(f"Wrote {len(ids)} tokens to {args.output}")
        else:
            space_delimited_tokens = ""
            for index, id in enumerate(ids):
                space_delimited_tokens += f"{id}"
                if index != len(ids) - 1:
                    space_delimited_tokens += " "
            print(space_delimited_tokens)


if __name__ == "__main__":
    main()
