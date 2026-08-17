"""
Regenerate include/unicode_ranges.hpp.

The Llama 3 pre-tokenizer regex uses \\p{L} and \\p{N}. The C++ tokenizer needs the
same character classes, so dump them from Python's unicodedata as sorted ranges.

Usage:
    python python/gen_unicode_ranges.py
"""

import unicodedata


def ranges(pred):
    out = []
    start = None
    for cp in range(0x110000):
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def fmt(rs, name):
    lines = [f"constexpr CodepointRange {name}[] = {{"]
    row = []
    for a, b in rs:
        row.append("{0x%X,0x%X}" % (a, b))
        if len(row) == 4:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    letters = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("L"))
    numbers = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("N"))

    header = f"""// GENERATED FILE - do not edit by hand.
// Unicode {unicodedata.unidata_version} general categories L* (letters) and N* (numbers),
// used by the Llama 3 pre-tokenizer regex (\\p{{L}} and \\p{{N}} classes).
// Regenerate with the snippet in python/gen_unicode_ranges.py

#pragma once

#include <cstdint>

struct CodepointRange
{{
    uint32_t first;
    uint32_t last;
}};

{fmt(letters, 'UNICODE_LETTER_RANGES')}

{fmt(numbers, 'UNICODE_NUMBER_RANGES')}
"""
    with open("include/unicode_ranges.hpp", "w") as f:
        f.write(header)
    print(f"letters: {len(letters)} ranges, numbers: {len(numbers)} ranges")


if __name__ == "__main__":
    main()
