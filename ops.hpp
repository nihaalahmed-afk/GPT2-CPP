#pragma once
#include <cmath>
#include <cstdint>


// LayerNorm Backward
void layernorm_backward(float* dinp, float* dw, float* db,
                        const float* dout, const float* inp, const float* w,
                        const float* mean, const float* rstd,
                        int B, int T, int C) {
    for (int bt = 0; bt < B * T; ++bt) {
        const float* dout_bt = dout + bt * C;
        const float* inp_bt  = inp  + bt * C;
        float* dinp_bt       = dinp + bt * C;

        float m = mean[bt];
        float s = rstd[bt];

        // Precompute two row-level dot products needed for input gradient
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int c = 0; c < C; ++c) {
            float norm = (inp_bt[c] - m) * s;
            float dnorm = dout_bt[c] * w[c];
            dnorm_mean += dnorm;
            dnorm_norm_mean += dnorm * norm;
        }
        dnorm_mean /= C;
        dnorm_norm_mean /= C;

        // Accumulate gradients for weights, biases, and inputs
        for (int c = 0; c < C; ++c) {
            float norm = (inp_bt[c] - m) * s;
            float dnorm = dout_bt[c] * w[c];

            db[c] += dout_bt[c];
            dw[c] += dout_bt[c] * norm;

            float dval = dnorm - dnorm_mean - norm * dnorm_norm_mean;
            dinp_bt[c] += dval * s;
        }
    }
}

// MatMul Forward: out = inp @ weight^T + bias
// inp:    [B * T, C_in]
// weight: [C_out, C_in]
// bias:   [C_out] (pass nullptr if no bias, e.g., lm_head)
// out:    [B * T, C_out]
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C_in, int C_out) {
    int BT = B * T;
    for (int bt = 0; bt < BT; ++bt) {
        const float* inp_row = inp + bt * C_in;
        float* out_row = out + bt * C_out;

        for (int o = 0; o < C_out; ++o) {
            const float* w_row = weight + o * C_in;
            float sum = (bias != nullptr) ? bias[o] : 0.0f;

            // Contiguous inner dot product (SIMD friendly)
            for (int i = 0; i < C_in; ++i) {
                sum += inp_row[i] * w_row[i];
            }
            out_row[o] = sum;
        }
    }
}


// MatMul Backward
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C_in, int C_out) {
    int BT = B * T;

    // 1. Backward into input (dinp) and bias (dbias)
    for (int bt = 0; bt < BT; ++bt) {
        const float* dout_row = dout + bt * C_out;
        float* dinp_row = dinp + bt * C_in;

        for (int o = 0; o < C_out; ++o) {
            float d = dout_row[o];
            if (dbias != nullptr) {
                dbias[o] += d;
            }
            const float* w_row = weight + o * C_in;
            for (int i = 0; i < C_in; ++i) {
                dinp_row[i] += d * w_row[i];
            }
        }
    }

    // 2. Backward into weights (dweight)
    for (int o = 0; o < C_out; ++o) {
        float* dw_row = dweight + o * C_in;
        for (int bt = 0; bt < BT; ++bt) {
            float d = dout[bt * C_out + o];
            const float* inp_row = inp + bt * C_in;
            for (int i = 0; i < C_in; ++i) {
                dw_row[i] += d * inp_row[i];
            }
        }
    }
}

// Causal Multi-Head Self-Attention Forward
// qkv:    [B, T, 3 * C] (input Queries, Keys, Values stacked)
// preatt: [B, NH, T, T] (raw attention scores before softmax - saved for backward)
// att:    [B, NH, T, T] (post-softmax attention weights - saved for backward)
// out:    [B, T, C]     (weighted sum of Values across all heads)
void attention_forward(float* out, float* preatt, float* att,
                       const float* qkv,
                       int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH; // Head size (e.g., 256 / 4 = 64)
    float scale = 1.0f / std::sqrt(static_cast<float>(hs));

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                const float* query = qkv + b * T * C3 + t * C3 + h * hs;
                float* preatt_row = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_row    = att    + b * NH * T * T + h * T * T + t * T;

                // Pass 1: Compute scaled Q . K^T for all past positions (t2 <= t)
                float maxval = -std::numeric_limits<float>::infinity();
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* key = qkv + b * T * C3 + t2 * C3 + C + h * hs;
                    float val = 0.0f;
                    for (int i = 0; i < hs; ++i) {
                        val += query[i] * key[i];
                    }
                    val *= scale;
                    if (val > maxval) maxval = val;
                    preatt_row[t2] = val;
                }

                // Pass 2: Numerically stable Softmax over t2 <= t
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; ++t2) {
                    float expv = std::exp(preatt_row[t2] - maxval);
                    expsum += expv;
                    att_row[t2] = expv;
                }
                float expsum_inv = (expsum == 0.0f) ? 0.0f : (1.0f / expsum);
                for (int t2 = 0; t2 <= t; ++t2) {
                    att_row[t2] *= expsum_inv;
                }
                // Zero out future masked positions (t2 > t)
                for (int t2 = t + 1; t2 < T; ++t2) {
                    att_row[t2] = 0.0f;
                }

                // Pass 3: Weighted sum of Values -> out[b, t, h*hs : (h+1)*hs]
                float* out_head = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; ++i) {
                    out_head[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* value = qkv + b * T * C3 + t2 * C3 + 2 * C + h * hs;
                    float a = att_row[t2];
                    for (int i = 0; i < hs; ++i) {
                        out_head[i] += a * value[i];
                    }
                }
            }
        }
    }
}

// Causal Multi-Head Self-Attention Backward
void attention_backward(float* dqkv, float* dpreatt, float* datt,
                        const float* dout, const float* qkv, const float* att,
                        int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / std::sqrt(static_cast<float>(hs));

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                const float* att_row    = att     + b * NH * T * T + h * T * T + t * T;
                float* datt_row         = datt    + b * NH * T * T + h * T * T + t * T;
                float* dpreatt_row      = dpreatt + b * NH * T * T + h * T * T + t * T;
                const float* query      = qkv     + b * T * C3 + t * C3 + h * hs;
                float* dquery           = dqkv    + b * T * C3 + t * C3 + h * hs;
                const float* dout_head  = dout    + b * T * C + t * C + h * hs;

                // Reverse Pass 3: Backward through Value accumulation
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* value = qkv  + b * T * C3 + t2 * C3 + 2 * C + h * hs;
                    float* dvalue      = dqkv + b * T * C3 + t2 * C3 + 2 * C + h * hs;
                    float a = att_row[t2];

                    for (int i = 0; i < hs; ++i) {
                        datt_row[t2] += value[i] * dout_head[i];
                        dvalue[i]    += a * dout_head[i];
                    }
                }

                // Reverse Pass 2: Backward through Softmax (t2 <= t)
                for (int t2 = 0; t2 <= t; ++t2) {
                    for (int t3 = 0; t3 <= t; ++t3) {
                        float indicator = (t2 == t3) ? 1.0f : 0.0f;
                        float local_deriv = att_row[t2] * (indicator - att_row[t3]);
                        dpreatt_row[t3] += local_deriv * datt_row[t2];
                    }
                }

                // Reverse Pass 1: Backward through scaled Q . K^T
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* key = qkv  + b * T * C3 + t2 * C3 + C + h * hs;
                    float* dkey      = dqkv + b * T * C3 + t2 * C3 + C + h * hs;
                    float d = dpreatt_row[t2] * scale;

                    for (int i = 0; i < hs; ++i) {
                        dquery[i] += key[i] * d;
                        dkey[i]   += query[i] * d;
                    }
                }
            }
        }
    }
}

// Residual Connection Forward: out = inp1 + inp2
void residual_forward(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = inp1[i] + inp2[i];
    }
}

// Residual Connection Backward
void residual_backward(float* dinp1, float* dinp2, const float* dout, int N) {
    for (int i = 0; i < N; ++i) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}


// GELU Forward (GPT-2 tanh approximation)
void gelu_forward(float* out, const float* inp, int N) {
    const float s = std::sqrt(2.0f / static_cast<float>(M_PI));
    for (int i = 0; i < N; ++i) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + std::tanh(s * (x + cube)));
    }
}

// GELU Backward
void gelu_backward(float* dinp, const float* inp, const float* dout, int N) {
    const float s = std::sqrt(2.0f / static_cast<float>(M_PI));
    for (int i = 0; i < N; ++i) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = s * (x + cube);
        float tanh_out = std::tanh(tanh_arg);
        float cosh_out = std::cosh(tanh_arg);
        float sech2 = 1.0f / (cosh_out * cosh_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech2 * s * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}


// Softmax + Cross-Entropy Loss Forward
// logits:  [B * T, V]
// targets: [B * T] uint16_t ground-truth token IDs
// probs:   [B * T, V] (saved for backward pass)
// losses:  [B * T]    (per-token loss)
void crossentropy_softmax_forward(float* probs, float* losses,
                                  const float* logits, const uint16_t* targets,
                                  int B, int T, int V) {
    for (int bt = 0; bt < B * T; ++bt) {
        const float* logits_row = logits + bt * V;
        float* probs_row = probs + bt * V;

        // 1. Find max logit for numerical stability
        float maxval = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < V; ++v) {
            if (logits_row[v] > maxval) maxval = logits_row[v];
        }

        // 2. Compute exp and sum
        float sum = 0.0f;
        for (int v = 0; v < V; ++v) {
            probs_row[v] = std::exp(logits_row[v] - maxval);
            sum += probs_row[v];
        }

        // 3. Normalize into probabilities and compute negative log-likelihood
        float inv_sum = 1.0f / sum;
        for (int v = 0; v < V; ++v) {
            probs_row[v] *= inv_sum;
        }

        int target_id = targets[bt];
        losses[bt] = -std::log(std::max(probs_row[target_id], 1e-10f));
    }
}

// Softmax + Cross-Entropy Backward
// dlogits: [B * T, V]
void crossentropy_softmax_backward(float* dlogits,
                                   const float* probs, const uint16_t* targets,
                                   int B, int T, int V) {
    float dloss = 1.0f / static_cast<float>(B * T);
    for (int bt = 0; bt < B * T; ++bt) {
        float* dlogits_row = dlogits + bt * V;
        const float* probs_row = probs + bt * V;
        int target_id = targets[bt];

        for (int v = 0; v < V; ++v) {
            float indicator = (v == target_id) ? 1.0f : 0.0f;
            dlogits_row[v] += (probs_row[v] - indicator) * dloss;
        }
    }
}
