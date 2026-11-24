#pragma once
#include "gpt2_params.hpp"
#include "ops.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Holds pointers to all intermediate forward activations (and their gradients)
struct ActivationTensors {
    float* encoded;    // [B, T, C]
    float* ln1;        // [L, B, T, C]
    float* ln1_mean;   // [L, B, T]
    float* ln1_rstd;   // [L, B, T]
    float* qkv;        // [L, B, T, 3*C]
    float* atty;       // [L, B, T, C]
    float* preatt;     // [L, B, NH, T, T]
    float* att;        // [L, B, NH, T, T]
    float* attproj;    // [L, B, T, C]
    float* residual2;  // [L, B, T, C]
    float* ln2;        // [L, B, T, C]
    float* ln2_mean;   // [L, B, T]
    float* ln2_rstd;   // [L, B, T]
    float* fch;        // [L, B, T, 4*C]
    float* fch_gelu;   // [L, B, T, 4*C]
    float* fcproj;     // [L, B, T, C]
    float* residual3;  // [L, B, T, C]
    float* lnf;        // [B, T, C]
    float* lnf_mean;   // [B, T]
    float* lnf_rstd;   // [B, T]
    float* logits;     // [B, T, V]
    float* probs;      // [B, T, V]
    float* losses;     // [B, T]
};

class GPT2 : public GPT2Model {
private:
    int alloc_B = 0;
    int alloc_T = 0;
    size_t num_activations = 0;

    std::vector<float> acts_memory;
    std::vector<float> grads_acts_memory;
    ActivationTensors acts;
    ActivationTensors grads_acts;

    // AdamW optimizer state buffers (1st and 2nd moments)
    std::vector<float> m_memory;
    std::vector<float> v_memory;
    std::vector<uint8_t> dropout_masks; // Stores masks for emb + (attn + mlp) * L
    std::mt19937 drop_rng{1337};

    int adam_step = 0;

    void allocate_activations(int B, int T) {
        if (B == alloc_B && T == alloc_T) return; // Already allocated for this shape
        alloc_B = B;
        alloc_T = T;

        size_t L  = cfg.n_layer;
        size_t NH = cfg.n_head;
        size_t C  = cfg.n_embd;
        size_t V  = cfg.vocab_size;
        dropout_masks.resize((1 + 2 * L) * B * T * C);


        size_t sizes[23] = {
            (size_t)B * T * C,           // encoded
            L * B * T * C,               // ln1
            L * B * T,                   // ln1_mean
            L * B * T,                   // ln1_rstd
            L * B * T * 3 * C,           // qkv
            L * B * T * C,               // atty
            L * B * NH * T * T,          // preatt
            L * B * NH * T * T,          // att
            L * B * T * C,               // attproj
            L * B * T * C,               // residual2
            L * B * T * C,               // ln2
            L * B * T,                   // ln2_mean
            L * B * T,                   // ln2_rstd
            L * B * T * 4 * C,           // fch
            L * B * T * 4 * C,           // fch_gelu
            L * B * T * C,               // fcproj
            L * B * T * C,               // residual3
            (size_t)B * T * C,           // lnf
            (size_t)B * T,               // lnf_mean
            (size_t)B * T,               // lnf_rstd
            (size_t)B * T * V,           // logits
            (size_t)B * T * V,           // probs
            (size_t)B * T                // losses
        };

        num_activations = 0;
        for (size_t s : sizes) num_activations += s;

        acts_memory.assign(num_activations, 0.0f);
        grads_acts_memory.assign(num_activations, 0.0f);

        assign_act_pointers(acts,       acts_memory.data(),       sizes);
        assign_act_pointers(grads_acts, grads_acts_memory.data(), sizes);
    }

    void assign_act_pointers(ActivationTensors& at, float* base, const size_t sizes[23]) {
        float** ptrs[23] = {
            &at.encoded, &at.ln1, &at.ln1_mean, &at.ln1_rstd, &at.qkv, &at.atty,
            &at.preatt, &at.att, &at.attproj, &at.residual2, &at.ln2, &at.ln2_mean,
            &at.ln2_rstd, &at.fch, &at.fch_gelu, &at.fcproj, &at.residual3,
            &at.lnf, &at.lnf_mean, &at.lnf_rstd, &at.logits, &at.probs, &at.losses
        };
        size_t offset = 0;
        for (int i = 0; i < 23; ++i) {
            *(ptrs[i]) = base + offset;
            offset += sizes[i];
        }
    }

public:
    float mean_loss = -1.0f;

    GPT2(GPT2Config config) : GPT2Model(config) {
        m_memory.resize(num_parameters, 0.0f);
        v_memory.resize(num_parameters, 0.0f);
    }

    // Forward Pass (pass targets = nullptr during inference/generation)
        float forward(const uint16_t* inputs, const uint16_t* targets, int B, int T, bool training = true) {
        allocate_activations(B, T);

        int L  = cfg.n_layer;
        int NH = cfg.n_head;
        int C  = cfg.n_embd;
        int V  = cfg.vocab_size;

        // 1. Token + Position Embeddings
        embedding_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
        uint8_t* emb_mask = dropout_masks.data();
        dropout_forward(acts.encoded, emb_mask, acts.encoded, B * T * C, cfg.dropout, training, drop_rng);



        // 2. Transformer Blocks (0 .. L-1)
        for (int l = 0; l < L; ++l) {
            float* residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

            // LayerNorm 1
            float* l_ln1      = acts.ln1      + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual,
                              params.ln1w + l * C, params.ln1b + l * C, B, T, C);

            // QKV Projection + Causal Multi-Head Attention + Output Projection
            float* l_qkv     = acts.qkv     + l * B * T * 3 * C;
            float* l_atty    = acts.atty    + l * B * T * C;
            float* l_preatt  = acts.preatt  + l * B * NH * T * T;
            float* l_att     = acts.att     + l * B * NH * T * T;
            float* l_attproj = acts.attproj + l * B * T * C;

            matmul_forward(l_qkv, l_ln1, params.qkvw + l * 3 * C * C, params.qkvb + l * 3 * C, B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, params.attprojw + l * C * C, params.attprojb + l * C, B, T, C, C);

            uint8_t* att_mask = dropout_masks.data() + (1 + 2 * l) * B * T * C;
            dropout_forward(l_attproj, att_mask, l_attproj, B * T * C, cfg.dropout, training, drop_rng);


            // Residual 2
            float* l_residual2 = acts.residual2 + l * B * T * C;
            residual_forward(l_residual2, residual, l_attproj, B * T * C);

            // LayerNorm 2 + MLP (C -> 4C -> GELU -> C)
            float* l_ln2      = acts.ln2      + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch      = acts.fch      + l * B * T * 4 * C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
            float* l_fcproj   = acts.fcproj   + l * B * T * C;

            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2,
                              params.ln2w + l * C, params.ln2b + l * C, B, T, C);
            matmul_forward(l_fch, l_ln2, params.fcw + l * 4 * C * C, params.fcb + l * 4 * C, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward(l_fcproj, l_fch_gelu, params.fcprojw + l * C * 4 * C, params.fcprojb + l * C, B, T, 4 * C, C);

            // Residual 3
            float* l_residual3 = acts.residual3 + l * B * T * C;
            uint8_t* mlp_mask = dropout_masks.data() + (1 + 2 * l + 1) * B * T * C;
            dropout_forward(l_fcproj, mlp_mask, l_fcproj, B * T * C, cfg.dropout, training, drop_rng);

            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }

        // 3. Final LayerNorm + Weight-Tied Output Logits (lm_head shares wte)
        float* last_residual = acts.residual3 + (L - 1) * B * T * C;
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, last_residual, params.lnfw, params.lnfb, B, T, C);
        matmul_forward(acts.logits, acts.lnf, params.wte, nullptr, B, T, C, V);

        // 4. Cross-Entropy Loss (only when targets are provided)
        if (targets != nullptr) {
            crossentropy_softmax_forward(acts.probs, acts.losses, acts.logits, targets, B, T, V);
            float total_loss = 0.0f;
            for (int i = 0; i < B * T; ++i) total_loss += acts.losses[i];
            mean_loss = total_loss / static_cast<float>(B * T);
        } else {
            mean_loss = -1.0f;
        }
        return mean_loss;
    }

    // Zero all parameter and activation gradient buffers before backward()
    void zero_grad() {
        std::fill(grads_memory.begin(), grads_memory.end(), 0.0f);
        std::fill(grads_acts_memory.begin(), grads_acts_memory.end(), 0.0f);
    }

    // Backward Pass (reverses forward() step-by-step)
    void backward(const uint16_t* inputs, const uint16_t* targets, int B, int T) {
        int L  = cfg.n_layer;
        int NH = cfg.n_head;
        int C  = cfg.n_embd;
        int V  = cfg.vocab_size;

        // 1. Backward through Cross-Entropy + Logits + Final LayerNorm
        crossentropy_softmax_backward(grads_acts.logits, acts.probs, targets, B, T, V);
        matmul_backward(grads_acts.lnf, grads.wte, nullptr, grads_acts.logits, acts.lnf, params.wte, B, T, C, V);

        float* last_residual  = acts.residual3       + (L - 1) * B * T * C;
        float* dlast_residual = grads_acts.residual3 + (L - 1) * B * T * C;
        layernorm_backward(dlast_residual, grads.lnfw, grads.lnfb,
                           grads_acts.lnf, last_residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

        // 2. Backward through Transformer Blocks in reverse (L-1 .. 0)
        for (int l = L - 1; l >= 0; --l) {
            float* residual  = (l == 0) ? acts.encoded       : acts.residual3       + (l - 1) * B * T * C;
            float* dresidual = (l == 0) ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;

            float* l_ln1       = acts.ln1       + l * B * T * C;
            float* l_qkv       = acts.qkv       + l * B * T * 3 * C;
            float* l_atty      = acts.atty      + l * B * T * C;
            float* l_att       = acts.att       + l * B * NH * T * T;
            float* l_residual2 = acts.residual2 + l * B * T * C;
            float* l_ln2       = acts.ln2       + l * B * T * C;
            float* l_fch       = acts.fch       + l * B * T * 4 * C;
            float* l_fch_gelu  = acts.fch_gelu  + l * B * T * 4 * C;

            float* dl_ln1       = grads_acts.ln1       + l * B * T * C;
            float* dl_qkv       = grads_acts.qkv       + l * B * T * 3 * C;
            float* dl_atty      = grads_acts.atty      + l * B * T * C;
            float* dl_preatt    = grads_acts.preatt    + l * B * NH * T * T;
            float* dl_att       = grads_acts.att       + l * B * NH * T * T;
            float* dl_attproj   = grads_acts.attproj   + l * B * T * C;
            float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
            float* dl_ln2       = grads_acts.ln2       + l * B * T * C;
            float* dl_fch       = grads_acts.fch       + l * B * T * 4 * C;
            float* dl_fch_gelu  = grads_acts.fch_gelu  + l * B * T * 4 * C;
            float* dl_fcproj    = grads_acts.fcproj    + l * B * T * C;
            float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

            // Reverse Residual 3 & MLP
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
            const uint8_t* mlp_mask = dropout_masks.data() + (1 + 2 * l + 1) * B * T * C;
           // Apply mask in-place to dl_fcproj before matmul_backward
            float scale = (cfg.dropout > 0.0f) ? (1.0f / (1.0f - cfg.dropout)) : 1.0f;
            for (int i = 0; i < B * T * C; ++i) dl_fcproj[i] *= (mlp_mask[i] ? scale : 0.0f);

            matmul_backward(dl_fch_gelu, grads.fcprojw + l * C * 4 * C, grads.fcprojb + l * C,
                            dl_fcproj, l_fch_gelu, params.fcprojw + l * C * 4 * C, B, T, 4 * C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, grads.fcw + l * 4 * C * C, grads.fcb + l * 4 * C,
                            dl_fch, l_ln2, params.fcw + l * 4 * C * C, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, grads.ln2w + l * C, grads.ln2b + l * C,
                               dl_ln2, l_residual2, params.ln2w + l * C,
                               acts.ln2_mean + l * B * T, acts.ln2_rstd + l * B * T, B, T, C);

            // Reverse Residual 2 & Attention
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
            const uint8_t* att_mask = dropout_masks.data() + (1 + 2 * l) * B * T * C;
            for (int i = 0; i < B * T * C; ++i) dl_attproj[i] *= (att_mask[i] ? scale : 0.0f);

            matmul_backward(dl_atty, grads.attprojw + l * C * C, grads.attprojb + l * C,
                            dl_attproj, l_atty, params.attprojw + l * C * C, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            matmul_backward(dl_ln1, grads.qkvw + l * 3 * C * C, grads.qkvb + l * 3 * C,
                            dl_qkv, l_ln1, params.qkvw + l * 3 * C * C, B, T, C, 3 * C);
            layernorm_backward(dresidual, grads.ln1w + l * C, grads.ln1b + l * C,
                               dl_ln1, residual, params.ln1w + l * C,
                               acts.ln1_mean + l * B * T, acts.ln1_rstd + l * B * T, B, T, C);
        }

        // 3. Backward through Embeddings
        const uint8_t* emb_mask = dropout_masks.data();
        float scale = (cfg.dropout > 0.0f) ? (1.0f / (1.0f - cfg.dropout)) : 1.0f;
        for (int i = 0; i < B * T * C; ++i) grads_acts.encoded[i] *= (emb_mask[i] ? scale : 0.0f);

        embedding_backward(grads.wte, grads.wpe, grads_acts.encoded, inputs, B, T, C);
    }

    // AdamW Optimizer Step
    void update(float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
                float eps = 1e-8f, float weight_decay = 0.01f) {
        adam_step++;
        float bias_corr1 = 1.0f - std::pow(beta1, adam_step);
        float bias_corr2 = 1.0f - std::pow(beta2, adam_step);

        for (size_t i = 0; i < num_parameters; ++i) {
            float g = grads_memory[i];
            float p = params_memory[i];

            m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * g;
            v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * g * g;

            float m_hat = m_memory[i] / bias_corr1;
            float v_hat = v_memory[i] / bias_corr2;

            // Decoupled weight decay + Adam update
            params_memory[i] = p - lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * p);
        }
    }

    // Save & Load Trained Model Weights
    void save_checkpoint(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(params_memory.data()), num_parameters * sizeof(float));
        std::cout << "Saved model checkpoint to " << filepath << "\n";
    }

    void load_checkpoint(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            std::cerr << "Error: Could not open " << filepath << "\n";
            return;
        }
        in.read(reinterpret_cast<char*>(params_memory.data()), num_parameters * sizeof(float));
        std::cout << "Loaded model checkpoint from " << filepath << "\n";
    }

    // Autoregressive Text Generation (Temperature Sampling)
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              float temperature = 0.7f, int top_k = 35, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::vector<int> context = prompt_ids;

        for (int step = 0; step < max_new_tokens; ++step) {
            // Crop context to block_size if sequence grows longer than T_max
            int start = std::max(0, static_cast<int>(context.size()) - cfg.block_size);
            int T_curr = static_cast<int>(context.size()) - start;

            std::vector<uint16_t> inp(T_curr);
            for (int t = 0; t < T_curr; ++t) {
                inp[t] = static_cast<uint16_t>(context[start + t]);
            }

            // Run forward pass with B = 1, T = T_curr
            forward(inp.data(), nullptr, 1, T_curr);

            // Grab logits at the last time step (T_curr - 1)
            const float* last_logits = acts.logits + (T_curr - 1) * cfg.vocab_size;

            // Greedy decoding shortcut if temperature <= 0
            if (temperature <= 0.0f) {
                int next_tok = static_cast<int>(
                    std::max_element(last_logits, last_logits + cfg.vocab_size) - last_logits
                );
                context.push_back(next_tok);
                if (next_tok == cfg.vocab_size - 1) break;
                continue;
            }

            // 1. Scale logits by temperature
            std::vector<float> scaled_logits(cfg.vocab_size);
            for (int v = 0; v < cfg.vocab_size; ++v) {
                scaled_logits[v] = last_logits[v] / temperature;
            }

            // 2. Top-K filtering: mask out tokens outside the top K logits
            float threshold = -1e30f;
            if (top_k > 0 && top_k < cfg.vocab_size) {
                std::vector<float> temp = scaled_logits;
                std::nth_element(temp.begin(),
                                 temp.begin() + (cfg.vocab_size - top_k),
                                 temp.end());
                threshold = temp[cfg.vocab_size - top_k];
            }

            // 3. Numerically stable Softmax over kept tokens
            float max_l = -1e30f;
            for (int v = 0; v < cfg.vocab_size; ++v) {
                if (scaled_logits[v] >= threshold && scaled_logits[v] > max_l) {
                    max_l = scaled_logits[v];
                }
            }

            std::vector<float> probs(cfg.vocab_size, 0.0f);
            float sum = 0.0f;
            for (int v = 0; v < cfg.vocab_size; ++v) {
                if (scaled_logits[v] >= threshold) {
                    probs[v] = std::exp(scaled_logits[v] - max_l);
                    sum += probs[v];
                }
            }
            for (int v = 0; v < cfg.vocab_size; ++v) probs[v] /= sum;

            // 4. Sample next token
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int next_tok = dist(rng);
            context.push_back(next_tok);

            // Stop early if <|endoftext|> (2047) is generated
            if (next_tok == cfg.vocab_size - 1) break;
        }
        return context;
    }

};
