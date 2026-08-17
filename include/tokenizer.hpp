// Byte-level BPE tokenizer for Llama 3.2, reading HuggingFace's tokenizer.json.
//
// Only the pieces Llama 3 actually uses are implemented: the tiktoken-style split
// regex, ByteLevel byte<->unicode mapping, BPE merges with ignore_merges, and the
// <|...|> special tokens. Everything else in tokenizer.json is ignored.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer
{
public:
    // Reads tokenizer.json. Returns false (and prints why) if it can't be used.
    bool load(const std::string &path);

    // Text -> token ids. Special tokens written literally in the text (e.g.
    // "<|eot_id|>") are recognized and mapped to their id instead of being split.
    std::vector<int> encode(const std::string &text) const;

    // Wraps a user message in the Llama 3 instruct chat template and encodes it,
    // so the model is prompted the same way transformers' apply_chat_template does.
    std::vector<int> encodeChatPrompt(const std::string &user_message) const;

    // Token ids -> text. With skip_special_tokens the <|...|> markers are dropped.
    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const;

    // Single token -> its raw bytes, for streaming output. A multi-byte character
    // split across two tokens comes out correct as long as pieces are printed in order.
    std::string decodeToken(int id, bool skip_special_tokens = true) const;

    bool isSpecial(int id) const;
    int vocabSize() const { return static_cast<int>(id_to_token_.size()); }

private:
    // One pre-tokenizer piece (already byte-encoded) -> ids, applying BPE merges.
    void encodePiece(const std::string &piece, std::vector<int> &out) const;
    // Splits text on the Llama 3 pre-tokenizer regex, appending byte offsets.
    void preTokenize(const std::string &text, std::vector<std::string> &pieces) const;
    // Longest special token starting at text[pos], or -1.
    int matchSpecialToken(const std::string &text, size_t pos, size_t &match_len) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<bool> is_special_;
    // BPE merge ranks, keyed by "left right" (byte-encoded tokens never contain a space)
    std::unordered_map<std::string, int> merge_ranks_;
    std::unordered_map<std::string, int> special_token_to_id_;
    size_t max_special_token_len_ = 0;
    bool ignore_merges_ = true;

    // ByteLevel alphabet: byte -> its (utf-8 encoded) stand-in character, and back
    std::string byte_to_unicode_[256];
    std::unordered_map<std::string, unsigned char> unicode_to_byte_;
};
