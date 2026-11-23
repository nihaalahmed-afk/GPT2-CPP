#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cctype>   // for std::isalpha, std::isdigit
#include <cstdio>   // for snprintf
#include <sstream>  // for std::stringstream


// Helper to pack two 16-bit token IDs into one 32-bit key
inline uint32_t pack_pair(int a, int b) {
    return (static_cast<uint32_t>(a) << 16) | static_cast<uint32_t>(b);
}

// Helper to unpack a 32-bit key back into two token IDs
inline std::pair<int, int> unpack_pair(uint32_t packed) {
    return {static_cast<int>(packed >> 16), static_cast<int>(packed & 0xFFFF)};
}

// Helper to safely display invisible/control bytes (like \n, \t, or raw hex)
std::string escape_bytes(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c >= 32 && c <= 126) out += static_cast<char>(c); // Printable ASCII
        else {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            out += buf;
        }
    }
    return out;
}


class BPETokenizer {
private:
    std::vector<std::string> vocab;
    // Maps packed pair -> new merged token ID (also acts as merge priority)
    std::unordered_map<uint32_t, int> merges;
    std::unordered_map<std::string, int> special_tokens;


    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> chunks;
        size_t i = 0;
        while (i < text.size()) {
            std::string chunk;
            unsigned char curr = static_cast<unsigned char>(text[i]);

            // 1. Attach a single leading space if followed by a non-whitespace character
            if (curr == ' ' && i + 1 < text.size() && text[i + 1] != ' ' && text[i + 1] != '\n' && text[i + 1] != '\r') {
                chunk += text[i++];
                curr = static_cast<unsigned char>(text[i]);
            }

            // 2. Group contiguous letters & apostrophes (e.g., " word", "'tis", "don't")
            if (std::isalpha(curr) || curr == '\'') {
                while (i < text.size()) {
                    unsigned char c = static_cast<unsigned char>(text[i]);
                    if (std::isalpha(c) || c == '\'') {
                        chunk += text[i++];
                    } else {
                        break;
                    }
                }
            }
            // 3. Group contiguous digits (e.g., " 123")
            else if (std::isdigit(curr)) {
                while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
                    chunk += text[i++];
                }
            }
            // 4. Group contiguous newlines or runs of multiple spaces
            else if (curr == '\n' || curr == '\r' || curr == ' ') {
                unsigned char ws = curr;
                while (i < text.size() && static_cast<unsigned char>(text[i]) == ws) {
                    chunk += text[i++];
                }
            }
            // 5. Isolate single punctuation/symbols (:, ., ,, ?, !)
            else {
                chunk += text[i++];
            }

            chunks.push_back(std::move(chunk));
        }
        return chunks;
    }

    // Count frequencies of adjacent pairs
    std::unordered_map<uint32_t, int> get_stats(const std::vector<int>& ids) const {
        std::unordered_map<uint32_t, int> counts;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            counts[pack_pair(ids[i], ids[i + 1])]++;
        }
        return counts;
    }

    // Replace all occurrences of target_pair with new_id
    std::vector<int> merge_tokens(const std::vector<int>& ids, uint32_t target_pair, int new_id) const {
        std::vector<int> new_ids;
        new_ids.reserve(ids.size()); // Prevent reallocations

        size_t i = 0;
        while (i < ids.size()) {
            if (i + 1 < ids.size() && pack_pair(ids[i], ids[i + 1]) == target_pair) {
                new_ids.push_back(new_id);
                i += 2;
            } else {
                new_ids.push_back(ids[i]);
                i += 1;
            }
        }
        return new_ids;
    }

public:
    BPETokenizer() {
        // Initialize base 256 byte vocabulary
        vocab.resize(256);
        for (int i = 0; i < 256; ++i) {
            vocab[i] = std::string(1, static_cast<char>(i));
        }
    }   
        void save(const std::string& filepath) const {
            std::ofstream out(filepath, std::ios::binary);
            uint32_t num_merges = static_cast<uint32_t>(merges.size());
            out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));

            std::vector<std::pair<int, uint32_t>> ordered;
            ordered.reserve(num_merges);
            for (const auto& [pair, id] : merges) {
                ordered.push_back({id, pair});
            }
            std::sort(ordered.begin(), ordered.end());

            for (const auto& [id, pair] : ordered) {
                out.write(reinterpret_cast<const char*>(&pair), sizeof(pair));
            }
    }

        void load(const std::string& filepath) {
            std::ifstream in(filepath, std::ios::binary);
            if (!in) {
                std::cerr << "Error: Could not open " << filepath << "\n";
                return;
            }
            uint32_t num_merges = 0;
            in.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));

            merges.clear();
            special_tokens.clear();
            vocab.resize(256 + num_merges);

            for (uint32_t i = 0; i < num_merges; ++i) {
                uint32_t pair = 0;
                in.read(reinterpret_cast<char*>(&pair), sizeof(pair));
                int id = 256 + i;
                merges[pair] = id;
                auto [p0, p1] = unpack_pair(pair);
                vocab[id] = vocab[p0] + vocab[p1];
            }

            // Restore <|endoftext|> at ID 2047
            int eot_id = static_cast<int>(vocab.size());
            vocab.push_back("<|endoftext|>");
            special_tokens["<|endoftext|>"] = eot_id;
        }


        void encode_file_to_bins(const std::string& input_txt,
                                    const std::string& train_bin,
                                    const std::string& val_bin,
                                    double train_split = 0.9) const {
                std::ifstream file(input_txt, std::ios::binary);
                if (!file) {
                    std::cerr << "Error: Could not open " << input_txt << "\n";
                    return;
                }
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::vector<int> all_ids = encode(buffer.str());

                // Pack into 16-bit unsigned integers (2 bytes per token)
                std::vector<uint16_t> tokens(all_ids.begin(), all_ids.end());
                size_t split_idx = static_cast<size_t>(tokens.size() * train_split);

                std::ofstream f_train(train_bin, std::ios::binary);
                f_train.write(reinterpret_cast<const char*>(tokens.data()), split_idx * sizeof(uint16_t));

                std::ofstream f_val(val_bin, std::ios::binary);
                f_val.write(reinterpret_cast<const char*>(tokens.data() + split_idx),
                            (tokens.size() - split_idx) * sizeof(uint16_t));

                std::cout << "Exported " << tokens.size() << " total tokens -> "
                        << "train.bin (" << split_idx << "), "
                        << "val.bin (" << (tokens.size() - split_idx) << ")\n";
            }


        void train(const std::string& filepath, int vocab_size = 512) {
            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                std::cerr << "Error: Could not open " << filepath << "\n";
                return;
            }

            // Read entire file into a string and pre-tokenize into chunks
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string text = buffer.str();
            file.close();

            std::vector<std::string> str_chunks = pre_tokenize(text);
            std::vector<std::vector<int>> chunk_ids;
            chunk_ids.reserve(str_chunks.size());

            for (const auto& s : str_chunks) {
                std::vector<int> ids;
                ids.reserve(s.size());
                for (unsigned char c : s) {
                    ids.push_back(c);
                }
                chunk_ids.push_back(std::move(ids));
            }

            std::cout << "Loaded " << text.size() << " bytes into " << chunk_ids.size() << " chunks.\n";

            int num_merges = vocab_size - 256;
            vocab.resize(vocab_size);

            for (int i = 0; i < num_merges; ++i) {
                // Count pairs strictly INSIDE each chunk (never across chunk boundaries)
                std::unordered_map<uint32_t, int> stats;
                for (const auto& ids : chunk_ids) {
                    for (size_t j = 0; j + 1 < ids.size(); ++j) {
                        stats[pack_pair(ids[j], ids[j + 1])]++;
                    }
                }
                if (stats.empty()) break;

                uint32_t top_pair = 0;
                int max_count = -1;
                for (const auto& [pair, count] : stats) {
                    if (count > max_count) {
                        max_count = count;
                        top_pair = pair;
                    }
                }

                int new_id = 256 + i;
                // Apply merge inside each chunk
                for (auto& ids : chunk_ids) {
                    if (ids.size() >= 2) {
                        ids = merge_tokens(ids, top_pair, new_id);
                    }
                }
                merges[top_pair] = new_id;

                auto [p0, p1] = unpack_pair(top_pair);
                vocab[new_id] = vocab[p0] + vocab[p1];
                std::cout << "Merge " << i + 1 << "/" << num_merges << " | "
                        << "ID " << new_id << ": "
                        << "[" << escape_bytes(vocab[p0]) << "] + [" << escape_bytes(vocab[p1]) << "] "
                        << "-> \"" << escape_bytes(vocab[new_id]) << "\" "
                        << "(count: " << max_count << ")\n";
            }
            int eot_id = static_cast<int>(vocab.size()); // 2047
            vocab.push_back("<|endoftext|>");
            special_tokens["<|endoftext|>"] = eot_id;
            std::cout << "Training complete. Vocab size: " << vocab.size() << "\n";
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> final_ids;
        const std::string eot = "<|endoftext|>";
        size_t pos = 0;

        while (pos < text.size()) {
            size_t found = text.find(eot, pos);
            std::string segment = (found == std::string::npos)
                                      ? text.substr(pos)
                                      : text.substr(pos, found - pos);

            for (const auto& chunk : pre_tokenize(segment)) {
                std::vector<int> ids;
                ids.reserve(chunk.size());
                for (unsigned char c : chunk) {
                    ids.push_back(c);
                }

                while (ids.size() >= 2) {
                    auto stats = get_stats(ids);
                    uint32_t best_pair = 0;
                    int lowest_merge_id = std::numeric_limits<int>::max();

                    for (const auto& [pair, count] : stats) {
                        auto it = merges.find(pair);
                        if (it != merges.end() && it->second < lowest_merge_id) {
                            lowest_merge_id = it->second;
                            best_pair = pair;
                        }
                    }

                    if (lowest_merge_id == std::numeric_limits<int>::max()) {
                        break;
                    }
                    ids = merge_tokens(ids, best_pair, lowest_merge_id);
                }
                final_ids.insert(final_ids.end(), ids.begin(), ids.end());
            }

            if (found == std::string::npos) break;
            final_ids.push_back(special_tokens.at(eot)); // Inserts ID 2047
            pos = found + eot.size();
        }
        return final_ids;
    }



    void print_vocab() const {
    std::cout << "\n=== FULL VOCABULARY (" << vocab.size() << " tokens) ===\n";
    for (size_t i = 0; i < vocab.size(); ++i) {
        std::cout << escape_bytes(vocab[i]) << " ";
    }
}


    std::string decode(const std::vector<int>& ids) const {
        std::string output;
        for (int id : ids) {
            if (id >= 0 && id < static_cast<int>(vocab.size())) {
                output += vocab[id];
            }
        }
        return output;
    }
};

int main() {
    BPETokenizer tokenizer;
    tokenizer.train("data.txt", 2047);
    tokenizer.save("tokenizer_2048.bin");

    tokenizer.print_vocab();
    std::cout<<"\n";
    BPETokenizer loaded_tok;
    loaded_tok.load("tokenizer_2048.bin");

    std::string test = "First Citizen:\nBefore we proceed.<|endoftext|>";
    auto ids = loaded_tok.encode(test);
    std::cout << "Round-trip match: " << (loaded_tok.decode(ids) == test ? "Yes" : "No")
              << " | Last token ID: " << ids.back() << "\n";

    // Export data.txt into flat uint16_t arrays for GPT-2 training
    loaded_tok.encode_file_to_bins("data.txt", "train.bin", "val.bin", 0.9);

    return 0;

}
