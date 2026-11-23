#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <stdexcept>

class DataLoader {
private:
    std::vector<uint16_t> tokens;
    int B; // Batch size
    int T; // Sequence length (block_size)
    size_t current_pos = 0;
    std::mt19937 rng;

public:
    std::vector<uint16_t> inputs;  // [B * T]
    std::vector<uint16_t> targets; // [B * T]

    DataLoader(const std::string& filepath, int batch_size, int seq_len, uint32_t seed = 42)
        : B(batch_size), T(seq_len), rng(seed) {
        
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("DataLoader error: Could not open " + filepath);
        }

        // Determine file size in bytes and resize uint16_t vector
        std::streamsize byte_size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t num_tokens = byte_size / sizeof(uint16_t);
        if (num_tokens < static_cast<size_t>(B * T + 1)) {
            throw std::runtime_error("DataLoader error: File too small for given B * T.");
        }

        tokens.resize(num_tokens);
        if (!file.read(reinterpret_cast<char*>(tokens.data()), byte_size)) {
            throw std::runtime_error("DataLoader error: Failed reading tokens from " + filepath);
        }

        inputs.resize(B * T);
        targets.resize(B * T);

        std::cout << "Loaded " << filepath << " (" << tokens.size() << " tokens, "
                  << num_batches() << " sequential batches per epoch)\n";
    }

    size_t num_tokens() const { return tokens.size(); }
    size_t num_batches() const { return (tokens.size() - 1) / (B * T); }

    void reset() { current_pos = 0; }

    // 1. Sequential Batching (Best for deterministic validation loss evaluation)
    void next_batch() {
        for (int b = 0; b < B; ++b) {
            // Wrap around if we reach the end of the dataset
            if (current_pos + T + 1 > tokens.size()) {
                current_pos = 0;
            }
            for (int t = 0; t < T; ++t) {
                inputs[b * T + t]  = tokens[current_pos + t];
                targets[b * T + t] = tokens[current_pos + t + 1];
            }
            current_pos += T;
        }
    }

    // 2. Random Batching (Best for training so batches aren't correlated)
    void random_batch() {
        std::uniform_int_distribution<size_t> dist(0, tokens.size() - T - 1);
        for (int b = 0; b < B; ++b) {
            size_t start_idx = dist(rng);
            for (int t = 0; t < T; ++t) {
                inputs[b * T + t]  = tokens[start_idx + t];
                targets[b * T + t] = tokens[start_idx + t + 1];
            }
        }
    }
};
