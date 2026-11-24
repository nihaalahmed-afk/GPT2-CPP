Hello, I am Nihaal.

I am going to try my best to implement a miniature GPT2 model in CPP.


# GPT2-CPP: GPT-2 From Scratch in Pure C++

A zero-dependency implementation of OpenAI's **GPT-2** Transformer architecture built completely from scratch in modern C++ (`C++17`), featuring custom forward/backward passes, a Byte-Pair Encoding (BPE) tokenizer, OpenMP multi-core parallelization, and autoregressive text generation with Top-K sampling.

No PyTorch, no TensorFlow, no BLAS/Eigen—just raw C++, contiguous memory arenas, and AVX2/OpenMP math kernels.

---

## Key Features

- **Pure C++ Implementation:** Custom matrix multiplication, fused QKV causal multi-head self-attention, LayerNorm, GELU, cross-entropy loss, and AdamW optimizer written from the ground up.
- **Contiguous Flat Memory Arenas:** All parameters (`params_memory`) and gradients (`grads_memory`) reside in two flat, cache-friendly `std::vector<float>` buffers with pointer-sliced tensor views (`ParameterTensors`).
- **Authentic GPT-2 Architecture:**
  - Pre-LayerNorm residual blocks (`LN -> Attention -> Residual -> LN -> MLP -> Residual`)
  - Weight tying between token embeddings (`wte`) and the final language modeling head
  - GPT-2 residual projection initialization scaling (\(1 / \sqrt{2L}\))
- **Multi-Core CPU Acceleration:** Accelerated with OpenMP (`-fopenmp`) and SIMD auto-vectorization (`-O3 -march=native`), achieving a **>2.3x speedup** on CPU training and **~55–78 tokens/sec** during inference.
- **Cosine LR Decay & Validation Checkpointing:** Built-in cosine learning rate annealing and automatic best-checkpoint saving (`gpt2_shakespeare_best.bin`).
- **Configurable Autoregressive Inference:** Supports temperature scaling, greedy decoding, and `std::nth_element` Top-K sampling.

---

## Project Structure

```text
GPT2-CPP/
├── gpt2.hpp             # GPT2Config, flat tensor memory management, model forward/backward, AdamW, generate()
├── ops.hpp              # Low-level math kernels (matmul, causal attention, LayerNorm, embeddings) with OpenMP
├── tokeniser.hpp        # BPE tokenizer encoder/decoder
├── dataloader.hpp       # Binary dataset batch loader (train.bin / val.bin)
├── train.cpp            # Training loop with cosine LR decay & validation loss evaluation
├── infer.cpp            # Interactive CLI text generation runner
└── tokenizer_2048.bin   # Pre-trained 2,048-token BPE vocabulary
```

---

## Model Configuration & Parameter Breakdown

The TinyShakespeare baseline run uses a lightweight, L3-cache-resident configuration (`1,071,872` parameters, ~4.3 MB FP32 weights):

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | `2048` | BPE vocabulary size |
| `block_size` | `128` | Maximum context window length (T) |
| `n_layer` | `4` | Number of Transformer blocks (L) |
| `n_head` | `4` | Number of attention heads (NH) |
| `n_embd` | `128` | Embedding dimension (C, `head_size = 32`) |
| **Total Parameters** | **`1,071,872`** | Weight-tied embeddings + 4 Transformer blocks |

*(Note: `GPT2Config` also supports scaling up to `n_embd = 256, block_size = 256` (~3.75M parameters) by adjusting `cfg` in `train.cpp`.)*

---

## Build & Run

### Prerequisites
- `g++` (GCC 9+) or `clang++` with OpenMP support
- Pre-tokenized `train.bin`, `val.bin`, and `tokenizer_2048.bin`

### 1. Compile Training & Inference Binaries
```bash
# Compile multi-core training engine
g++ -O3 -march=native -fopenmp train.cpp -o fast_train

# Compile interactive inference runner
g++ -O3 -march=native -fopenmp generate.cpp -o generate
```

### 2. Train the Model
```bash
./fast_train
```
- **Dataset:** TinyShakespeare (`342,667` train tokens / `38,075` validation tokens)
- **Batch Size:** `B = 8`, `T = 128` (`1,024` tokens/step)
- **Schedule:** Cosine decay from `1e-3` to `1e-4` over `3,000` steps (~9 epochs)

### 3. Run Interactive Inference
```bash
./generate
```

---

## Training Benchmarks & Results

### Loss Progression (TinyShakespeare, 3,000 Steps)
- **Initial Loss (Step 1):** `7.65771` (matches theoretical random-init baseline \(\ln(2048) \approx 7.6246\))
- **Best Validation Loss (Step 2600):** **`4.32868`** (Train Loss: `3.74201`) — automatically saved to `gpt2_shakespeare_best.bin`

| Step | Train Loss | Val Loss | Learning Rate | Checkpoint |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 7.6577 | 7.3931 | `0.00100` | Saved |
| 400 | 4.8391 | 4.9430 | `0.00096` | Saved |
| 1000 | 4.4700 | 4.6250 | `0.00078` | Saved |
| 1800 | 3.9353 | 4.3889 | `0.00041` | Saved |
| **2600** | **3.7420** | **4.3287** | **`0.00014`** | **Saved (Best)** |
| 3000 | 3.8450 | 4.3553 | `0.00010` | — |

### CPU Performance

| Optimization Level | Training Step Time | Inference Speed |
| :--- | :---: | :---: |
| Single-threaded baseline | ~595 ms/step | ~48 tokens/sec |
| **OpenMP + `-O3 -march=native`** | **~252 ms/step (2.36x faster)** | **~55–78 tokens/sec** |

---

## Sample Generations (`temperature = 0.7`, `top_k = 35`)

**Prompt:**
```text
KING RICHARD III:
Now is the winter of our discontent
```

**Generated Output:**
```text
KING RICHARD III:
Now is the winter of our discontent
Of the two seal?

KING RICHARD III:
An These fair ale, that injured to the greater metlealit,
From by the tony of his complst, and that you,
Your hearts always in his majesty.

KING RICHARD III:
And, tent and and, and Sicerved my father's
```

**Prompt:**
```text
POET:
```

**Generated Output:**
```text
POET:
Villain, that the which he beamk measure
with my head!
For my sides, the name of justice of sweetness.
Unest is, and I can do again,
And yet I that I; then to me hither my let'd, and heavens,
Were so I have thrinking to--port'd thee.
```

---

## Future Roadmap

- [ ] **KV-Caching:** Cache per-layer Key and Value projections during `generate()` to reduce autoregressive decoding from $O(T^2)$ to $O(T)$.
- [ ] **Parallelize `attention_backward` & `AdamW`:** Add OpenMP pragmas across `(B, NH)` in `attention_backward` and parameter updates to push step times under `100 ms`.
- [ ] **Gradient Norm Clipping:** Add global L2 norm clipping (`max_norm = 1.0f`) before optimizer steps for deeper (`6L–12L`) configurations.
- [ ] **Character-Level Tokenizer Option:** Add a 65-character vocabulary mode alongside BPE for crisp spelling on small datasets.

---
