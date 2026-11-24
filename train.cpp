#include "tokeniser.hpp"
#include "dataloader.hpp"
#include "gpt2.hpp"
#include <chrono>
#include <cmath>
#include <iostream>


float evaluate_val(GPT2& model, DataLoader& val_loader, int B, int T, int eval_batches = 20) {
    float total_loss = 0.0f;
    for (int i = 0; i < eval_batches; ++i) {
        val_loader.random_batch();
        total_loss += model.forward(val_loader.inputs.data(), val_loader.targets.data(), B, T);
    }
    return total_loss / eval_batches;
}

// Cosine decay learning rate schedule
float get_lr(int step, int total_steps, float max_lr = 1e-3f, float min_lr = 1e-4f) {
    float progress = static_cast<float>(step) / static_cast<float>(total_steps);
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(3.14159265f * progress));
}

int main() {
    BPETokenizer tokenizer;
    tokenizer.load("tokenizer_2048.bin");

    GPT2Config cfg;
    cfg.vocab_size = 2048;
    cfg.block_size = 128;
    cfg.n_layer    = 4;
    cfg.n_head     = 4;
    cfg.n_embd     = 128;

    GPT2 model(cfg);
    std::cout << "Model initialized with " << model.num_parameters << " parameters.\n";

    int B = 8;
    DataLoader train_loader("train.bin", B, cfg.block_size);
    DataLoader val_loader("val.bin", B, cfg.block_size);

    float best_val_loss = 1e9f;
    int total_steps = 3000; // Safe to try 3000 once LR decay + best-checkpoint saving are active

    
    for (int step = 1; step <= total_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();

        float lr = get_lr(step, total_steps);

        train_loader.random_batch();
        model.zero_grad();
        float loss = model.forward(train_loader.inputs.data(), train_loader.targets.data(), B, cfg.block_size);
        model.backward(train_loader.inputs.data(), train_loader.targets.data(), B, cfg.block_size);
        model.update(lr);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (step % 200 == 0 || step == 1) {
            float val_loss = evaluate_val(model, val_loader, B, cfg.block_size);
            std::cout << "Step " << step << "/" << total_steps
                      << " | Train Loss: " << loss
                      << " | Val Loss: " << val_loss
                      << " | LR: " << lr
                      << " | Time: " << ms << " ms/step\n";

            // Save only when validation loss improves
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                model.save_checkpoint("gpt2_shakespeare_best.bin");
            }
        }
    }

    std::cout << "\nBest Val Loss: " << best_val_loss << "\n";

    std::string prompt = "First Citizen:\n";
    auto prompt_ids = tokenizer.encode(prompt);
    auto generated_ids = model.generate(prompt_ids, 60, 0.8f);

    std::cout << "\n=== GENERATED SHAKESPEARE ===\n";
    std::cout << tokenizer.decode(generated_ids) << "\n";

    return 0;
}