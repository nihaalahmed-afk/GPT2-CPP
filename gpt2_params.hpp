#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstddef>

struct GPT2Config {
    int vocab_size = 2048;  // Matches your tokenizer_2048.bin
    int block_size = 256;   // Max context length (T)
    int n_layer    = 4;     // Number of Transformer blocks (L)
    int n_head     = 4;     // Number of attention heads (NH)
    int n_embd     = 256;   // Embedding dimension (C), head_size = C / NH = 64
    float dropout = 0.1;
};

// Holds raw pointers into a single flat memory buffer
struct ParameterTensors {
    float* wte;      // [V, C]
    float* wpe;      // [T_max, C]
    float* ln1w;     // [L, C]
    float* ln1b;     // [L, C]
    float* qkvw;     // [L, 3*C, C]
    float* qkvb;     // [L, 3*C]
    float* attprojw; // [L, C, C]
    float* attprojb; // [L, C]
    float* ln2w;     // [L, C]
    float* ln2b;     // [L, C]
    float* fcw;      // [L, 4*C, C]
    float* fcb;      // [L, 4*C]
    float* fcprojw;  // [L, C, 4*C]
    float* fcprojb;  // [L, C]
    float* lnfw;     // [C]
    float* lnfb;     // [C]
};

class GPT2Model {
public:
    GPT2Config cfg;
    size_t num_parameters = 0;

    std::vector<float> params_memory; // Flat storage for weights
    std::vector<float> grads_memory;  // Flat storage for gradients (same exact layout!)
    ParameterTensors params;
    ParameterTensors grads;

    GPT2Model(GPT2Config config) : cfg(config) {
        size_t V = cfg.vocab_size;
        size_t T = cfg.block_size;
        size_t L = cfg.n_layer;
        size_t C = cfg.n_embd;

        // Calculate exact sizes for all 16 parameter tensors
        size_t sizes[16] = {
            V * C,          // wte
            T * C,          // wpe
            L * C,          // ln1w
            L * C,          // ln1b
            L * 3 * C * C,  // qkvw
            L * 3 * C,      // qkvb
            L * C * C,      // attprojw
            L * C,          // attprojb
            L * C,          // ln2w
            L * C,          // ln2b
            L * 4 * C * C,  // fcw
            L * 4 * C,      // fcb
            L * C * 4 * C,  // fcprojw
            L * C,          // fcprojb
            C,              // lnfw
            C               // lnfb
        };

        for (size_t s : sizes) num_parameters += s;

        // Allocate contiguous memory for weights and gradients
        params_memory.resize(num_parameters, 0.0f);
        grads_memory.resize(num_parameters, 0.0f);

        // Point struct members to their offsets
        assign_pointers(params, params_memory.data(), sizes);
        assign_pointers(grads,  grads_memory.data(),  sizes);

        init_weights();
    }

private:
    void assign_pointers(ParameterTensors& pt, float* base, const size_t sizes[16]) {
        float** ptrs[16] = {
            &pt.wte, &pt.wpe, &pt.ln1w, &pt.ln1b, &pt.qkvw, &pt.qkvb,
            &pt.attprojw, &pt.attprojb, &pt.ln2w, &pt.ln2b, &pt.fcw, &pt.fcb,
            &pt.fcprojw, &pt.fcprojb, &pt.lnfw, &pt.lnfb
        };
        size_t offset = 0;
        for (int i = 0; i < 16; ++i) {
            *(ptrs[i]) = base + offset;
            offset += sizes[i];
        }
    }

    void init_weights(uint32_t seed = 1337) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.02f);

        // Initialize all weights with N(0, 0.02)
        for (size_t i = 0; i < num_parameters; ++i) {
            params_memory[i] = dist(rng);
        }

        // LayerNorm weights start at 1.0, all biases start at 0.0
        size_t L = cfg.n_layer;
        size_t C = cfg.n_embd;
        for (size_t i = 0; i < L * C; ++i) {
            params.ln1w[i] = 1.0f;
            params.ln1b[i] = 0.0f;
            params.ln2w[i] = 1.0f;
            params.ln2b[i] = 0.0f;
        }
        for (size_t i = 0; i < C; ++i) {
            params.lnfw[i] = 1.0f;
            params.lnfb[i] = 0.0f;
        }
        for (size_t i = 0; i < L * 3 * C; ++i) params.qkvb[i] = 0.0f;
        for (size_t i = 0; i < L * C; ++i)     params.attprojb[i] = 0.0f;
        for (size_t i = 0; i < L * 4 * C; ++i) params.fcb[i] = 0.0f;
        for (size_t i = 0; i < L * C; ++i)     params.fcprojb[i] = 0.0f;

        // GPT-2 residual scaling: scale residual projections by 1 / sqrt(2 * n_layer)
        float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(L));
        for (size_t i = 0; i < L * C * C; ++i)     params.attprojw[i] *= residual_scale;
        for (size_t i = 0; i < L * C * 4 * C; ++i) params.fcprojw[i] *= residual_scale;
    }
};
