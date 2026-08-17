#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "unicode_ranges.hpp"

using json = nlohmann::json;

namespace
{

// ---------------------------------------------------------------- utf-8 helpers

struct Codepoint
{
    uint32_t value;
    size_t length; // in bytes
};

// Decodes one codepoint at `pos`. Invalid bytes are returned as-is (length 1) so a
// malformed prompt still tokenizes into something instead of hanging.
Codepoint decodeUtf8(const std::string &text, size_t pos)
{
    unsigned char first = static_cast<unsigned char>(text[pos]);
    size_t length = 1;
    uint32_t value = first;
    if (first >= 0xF0)
    {
        length = 4;
        value = first & 0x07;
    }
    else if (first >= 0xE0)
    {
        length = 3;
        value = first & 0x0F;
    }
    else if (first >= 0xC0)
    {
        length = 2;
        value = first & 0x1F;
    }
    if (pos + length > text.size())
    {
        return {first, 1};
    }
    for (size_t i = 1; i < length; ++i)
    {
        unsigned char continuation = static_cast<unsigned char>(text[pos + i]);
        if ((continuation & 0xC0) != 0x80)
        {
            return {first, 1};
        }
        value = (value << 6) | (continuation & 0x3F);
    }
    return {value, length};
}

void appendUtf8(std::string &out, uint32_t codepoint)
{
    if (codepoint < 0x80)
    {
        out += static_cast<char>(codepoint);
    }
    else if (codepoint < 0x800)
    {
        out += static_cast<char>(0xC0 | (codepoint >> 6));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    else if (codepoint < 0x10000)
    {
        out += static_cast<char>(0xE0 | (codepoint >> 12));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    else
    {
        out += static_cast<char>(0xF0 | (codepoint >> 18));
        out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
}

// ------------------------------------------------------- unicode character classes

template <size_t N>
bool inRanges(uint32_t codepoint, const CodepointRange (&ranges)[N])
{
    size_t low = 0;
    size_t high = N;
    while (low < high)
    {
        size_t mid = (low + high) / 2;
        if (codepoint < ranges[mid].first)
        {
            high = mid;
        }
        else if (codepoint > ranges[mid].last)
        {
            low = mid + 1;
        }
        else
        {
            return true;
        }
    }
    return false;
}

bool isLetter(uint32_t codepoint) { return inRanges(codepoint, UNICODE_LETTER_RANGES); }
bool isNumber(uint32_t codepoint) { return inRanges(codepoint, UNICODE_NUMBER_RANGES); }

// \s in the Rust regex crate is the Unicode White_Space property
bool isSpace(uint32_t codepoint)
{
    return (codepoint >= 0x09 && codepoint <= 0x0D) || codepoint == 0x20 || codepoint == 0x85 ||
           codepoint == 0xA0 || codepoint == 0x1680 || (codepoint >= 0x2000 && codepoint <= 0x200A) ||
           codepoint == 0x2028 || codepoint == 0x2029 || codepoint == 0x202F || codepoint == 0x205F ||
           codepoint == 0x3000;
}

bool isNewline(uint32_t codepoint) { return codepoint == '\r' || codepoint == '\n'; }

// ---------------------------------------------------- llama 3 pre-tokenizer regex
//
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Hand-rolled because std::regex has no \p{L}/\p{N}. Each helper is one alternative
// and returns the match length at `pos` (0 = no match); they are tried in source
// order, which is what a backtracking engine does for alternations.

size_t matchContraction(const std::string &text, size_t pos)
{
    if (text[pos] != '\'' || pos + 1 >= text.size())
    {
        return 0;
    }
    char next = static_cast<char>(std::tolower(static_cast<unsigned char>(text[pos + 1])));
    char after = pos + 2 < text.size() ? static_cast<char>(std::tolower(static_cast<unsigned char>(text[pos + 2]))) : '\0';
    switch (next)
    {
    case 's':
    case 't':
    case 'm':
    case 'd':
        return 2;
    case 'r':
        return after == 'e' ? 3 : 0;
    case 'v':
        return after == 'e' ? 3 : 0;
    case 'l':
        return after == 'l' ? 3 : 0;
    default:
        return 0;
    }
}

// [^\r\n\p{L}\p{N}]?\p{L}+
size_t matchWord(const std::string &text, size_t pos)
{
    size_t cursor = pos;
    Codepoint first = decodeUtf8(text, cursor);
    if (!isLetter(first.value) && !isNumber(first.value) && !isNewline(first.value))
    {
        cursor += first.length; // the optional leading character (usually a space)
    }
    size_t letters_start = cursor;
    while (cursor < text.size())
    {
        Codepoint current = decodeUtf8(text, cursor);
        if (!isLetter(current.value))
        {
            break;
        }
        cursor += current.length;
    }
    // \p{L}+ needs at least one letter; backtracking the optional character can't help
    // because it only ever consumes a non-letter.
    return cursor == letters_start ? 0 : cursor - pos;
}

// \p{N}{1,3}
size_t matchDigits(const std::string &text, size_t pos)
{
    size_t cursor = pos;
    for (int count = 0; count < 3 && cursor < text.size(); ++count)
    {
        Codepoint current = decodeUtf8(text, cursor);
        if (!isNumber(current.value))
        {
            break;
        }
        cursor += current.length;
    }
    return cursor - pos;
}

//  ?[^\s\p{L}\p{N}]+[\r\n]*
size_t matchPunctuation(const std::string &text, size_t pos)
{
    size_t cursor = pos;
    if (text[cursor] == ' ')
    {
        ++cursor;
    }
    size_t body_start = cursor;
    while (cursor < text.size())
    {
        Codepoint current = decodeUtf8(text, cursor);
        if (isSpace(current.value) || isLetter(current.value) || isNumber(current.value))
        {
            break;
        }
        cursor += current.length;
    }
    if (cursor == body_start)
    {
        return 0; // backtracking the optional space is pointless, a space is \s
    }
    while (cursor < text.size() && isNewline(static_cast<unsigned char>(text[cursor])))
    {
        ++cursor;
    }
    return cursor - pos;
}

// End of the run of whitespace starting at `pos`, plus the start offset of its last
// codepoint (needed by the \s+(?!\S) alternative).
size_t whitespaceRunEnd(const std::string &text, size_t pos, size_t &last_codepoint_start)
{
    size_t cursor = pos;
    last_codepoint_start = pos;
    while (cursor < text.size())
    {
        Codepoint current = decodeUtf8(text, cursor);
        if (!isSpace(current.value))
        {
            break;
        }
        last_codepoint_start = cursor;
        cursor += current.length;
    }
    return cursor;
}

// \s*[\r\n]+
size_t matchTrailingNewlines(const std::string &text, size_t pos)
{
    size_t last_start = 0;
    size_t run_end = whitespaceRunEnd(text, pos, last_start);
    if (run_end == pos)
    {
        return 0;
    }
    // Greedy \s* backtracks to the last \r or \n in the run; [\r\n]+ then eats to the
    // end of that newline run, which is the character itself since nothing after it
    // in the whitespace run is a newline.
    for (size_t i = run_end; i > pos; --i)
    {
        if (isNewline(static_cast<unsigned char>(text[i - 1])))
        {
            return i - pos;
        }
    }
    return 0;
}

// \s+(?!\S)
size_t matchWhitespaceBeforeSpace(const std::string &text, size_t pos)
{
    size_t last_start = 0;
    size_t run_end = whitespaceRunEnd(text, pos, last_start);
    if (run_end == pos)
    {
        return 0;
    }
    if (run_end == text.size())
    {
        return run_end - pos; // nothing follows, the lookahead holds
    }
    // A non-space follows, so give the last whitespace character back to the next match.
    return last_start == pos ? 0 : last_start - pos;
}

// \s+
size_t matchWhitespace(const std::string &text, size_t pos)
{
    size_t last_start = 0;
    return whitespaceRunEnd(text, pos, last_start) - pos;
}

} // namespace

void Tokenizer::preTokenize(const std::string &text, std::vector<std::string> &pieces) const
{
    size_t pos = 0;
    while (pos < text.size())
    {
        size_t length = matchContraction(text, pos);
        if (length == 0)
        {
            length = matchWord(text, pos);
        }
        if (length == 0)
        {
            length = matchDigits(text, pos);
        }
        if (length == 0)
        {
            length = matchPunctuation(text, pos);
        }
        if (length == 0)
        {
            length = matchTrailingNewlines(text, pos);
        }
        if (length == 0)
        {
            length = matchWhitespaceBeforeSpace(text, pos);
        }
        if (length == 0)
        {
            length = matchWhitespace(text, pos);
        }
        if (length == 0)
        {
            length = decodeUtf8(text, pos).length; // unreachable for valid utf-8, but never loop forever
        }

        // ByteLevel(use_regex=false, add_prefix_space=false): every byte becomes its
        // stand-in character, which is the alphabet the vocab is written in.
        std::string piece;
        piece.reserve(length * 2);
        for (size_t i = 0; i < length; ++i)
        {
            piece += byte_to_unicode_[static_cast<unsigned char>(text[pos + i])];
        }
        pieces.push_back(std::move(piece));
        pos += length;
    }
}

bool Tokenizer::load(const std::string &path)
{
    std::ifstream tokenizer_file(path);
    if (!tokenizer_file.is_open())
    {
        std::cout << "Can't open " << path << " (download it with ./download_model.sh)\n";
        return false;
    }

    json tokenizer_json;
    try
    {
        tokenizer_file >> tokenizer_json;
    }
    catch (const json::exception &error)
    {
        std::cout << "Can't parse " << path << ": " << error.what() << "\n";
        return false;
    }

    // ByteLevel alphabet, same construction as GPT-2's bytes_to_unicode()
    bool printable[256] = {false};
    for (int byte = '!'; byte <= '~'; ++byte)
    {
        printable[byte] = true;
    }
    for (int byte = 0xA1; byte <= 0xAC; ++byte)
    {
        printable[byte] = true;
    }
    for (int byte = 0xAE; byte <= 0xFF; ++byte)
    {
        printable[byte] = true;
    }
    uint32_t next_free_codepoint = 256;
    for (int byte = 0; byte < 256; ++byte)
    {
        uint32_t codepoint = printable[byte] ? static_cast<uint32_t>(byte) : next_free_codepoint++;
        std::string encoded;
        appendUtf8(encoded, codepoint);
        byte_to_unicode_[byte] = encoded;
        unicode_to_byte_[encoded] = static_cast<unsigned char>(byte);
    }

    const json &model = tokenizer_json.at("model");
    if (model.contains("ignore_merges"))
    {
        ignore_merges_ = model.at("ignore_merges").get<bool>();
    }

    const json &vocab = model.at("vocab");
    token_to_id_.reserve(vocab.size() * 2);
    int max_id = 0;
    for (auto &[token, id_value] : vocab.items())
    {
        int id = id_value.get<int>();
        token_to_id_[token] = id;
        max_id = std::max(max_id, id);
    }

    const json &added_tokens = tokenizer_json.contains("added_tokens") ? tokenizer_json.at("added_tokens") : json::array();
    for (const json &added : added_tokens)
    {
        max_id = std::max(max_id, added.at("id").get<int>());
    }

    id_to_token_.assign(max_id + 1, std::string());
    is_special_.assign(max_id + 1, false);
    for (const auto &[token, id] : token_to_id_)
    {
        id_to_token_[id] = token;
    }

    for (const json &added : added_tokens)
    {
        int id = added.at("id").get<int>();
        std::string content = added.at("content").get<std::string>();
        token_to_id_[content] = id;
        id_to_token_[id] = content;
        bool special = !added.contains("special") || added.at("special").get<bool>();
        is_special_[id] = special;
        if (special)
        {
            special_token_to_id_[content] = id;
            max_special_token_len_ = std::max(max_special_token_len_, content.size());
        }
    }

    // merges are "left right" in the classic format, ["left", "right"] in newer dumps
    const json &merges = model.at("merges");
    merge_ranks_.reserve(merges.size() * 2);
    int rank = 0;
    for (const json &merge : merges)
    {
        std::string key;
        if (merge.is_array())
        {
            key = merge.at(0).get<std::string>() + " " + merge.at(1).get<std::string>();
        }
        else
        {
            key = merge.get<std::string>();
        }
        merge_ranks_.emplace(std::move(key), rank++);
    }

    std::cout << "Tokenizer loaded: " << id_to_token_.size() << " tokens, " << merge_ranks_.size()
              << " merges, " << special_token_to_id_.size() << " special tokens\n";
    return true;
}

void Tokenizer::encodePiece(const std::string &piece, std::vector<int> &out) const
{
    if (ignore_merges_)
    {
        auto whole = token_to_id_.find(piece);
        if (whole != token_to_id_.end())
        {
            out.push_back(whole->second);
            return;
        }
    }

    // start from single characters of the byte-level alphabet, then merge greedily by rank
    std::vector<std::string> symbols;
    for (size_t pos = 0; pos < piece.size();)
    {
        size_t length = decodeUtf8(piece, pos).length;
        symbols.emplace_back(piece, pos, length);
        pos += length;
    }

    while (symbols.size() > 1)
    {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_index = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i)
        {
            auto merge = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (merge != merge_ranks_.end() && merge->second < best_rank)
            {
                best_rank = merge->second;
                best_index = i;
            }
        }
        if (best_index == symbols.size())
        {
            break;
        }
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + best_index + 1);
    }

    for (const std::string &symbol : symbols)
    {
        auto id = token_to_id_.find(symbol);
        if (id != token_to_id_.end())
        {
            out.push_back(id->second);
        }
        else
        {
            std::cout << "Tokenizer: no vocab entry for \"" << symbol << "\", dropping it\n";
        }
    }
}

int Tokenizer::matchSpecialToken(const std::string &text, size_t pos, size_t &match_len) const
{
    if (text[pos] != '<')
    {
        return -1; // every Llama 3 special token looks like <|...|>
    }
    size_t longest = std::min(max_special_token_len_, text.size() - pos);
    for (size_t length = longest; length >= 3; --length)
    {
        auto special = special_token_to_id_.find(text.substr(pos, length));
        if (special != special_token_to_id_.end())
        {
            match_len = length;
            return special->second;
        }
    }
    return -1;
}

std::vector<int> Tokenizer::encode(const std::string &text) const
{
    std::vector<int> ids;
    std::vector<std::string> pieces;
    size_t segment_start = 0;
    for (size_t pos = 0; pos < text.size();)
    {
        size_t match_len = 0;
        int special_id = matchSpecialToken(text, pos, match_len);
        if (special_id < 0)
        {
            ++pos;
            continue;
        }
        pieces.clear();
        preTokenize(text.substr(segment_start, pos - segment_start), pieces);
        for (const std::string &piece : pieces)
        {
            encodePiece(piece, ids);
        }
        ids.push_back(special_id);
        pos += match_len;
        segment_start = pos;
    }

    pieces.clear();
    preTokenize(text.substr(segment_start), pieces);
    for (const std::string &piece : pieces)
    {
        encodePiece(piece, ids);
    }
    return ids;
}

std::vector<int> Tokenizer::encodeChatPrompt(const std::string &user_message) const
{
    // Llama 3 instruct format, without a system block - same shape as the token ids
    // this repo used to hardcode.
    return encode("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + user_message +
                  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
}

std::string Tokenizer::decodeToken(int id, bool skip_special_tokens) const
{
    if (id < 0 || id >= static_cast<int>(id_to_token_.size()))
    {
        return std::string();
    }
    if (skip_special_tokens && is_special_[id])
    {
        return std::string();
    }
    if (is_special_[id])
    {
        return id_to_token_[id]; // special tokens are literal text, not byte-encoded
    }

    const std::string &token = id_to_token_[id];
    std::string text;
    text.reserve(token.size());
    for (size_t pos = 0; pos < token.size();)
    {
        size_t length = decodeUtf8(token, pos).length;
        auto byte = unicode_to_byte_.find(token.substr(pos, length));
        if (byte != unicode_to_byte_.end())
        {
            text += static_cast<char>(byte->second);
        }
        pos += length;
    }
    return text;
}

std::string Tokenizer::decode(const std::vector<int> &ids, bool skip_special_tokens) const
{
    std::string text;
    for (int id : ids)
    {
        text += decodeToken(id, skip_special_tokens);
    }
    return text;
}

bool Tokenizer::isSpecial(int id) const
{
    return id >= 0 && id < static_cast<int>(is_special_.size()) && is_special_[id];
}
