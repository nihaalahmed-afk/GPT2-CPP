#include "tokeniser.hpp"
#include "dataloader.hpp"
#include "gpt2.hpp"
#include <chrono>

int main() {
    BPETokenizer tokenizer;
    tokenizer.load("tokenizer_2048.bin");

    // Small, fast CPU-friendly GPT-2 config (~1.6M params)
    GPT2Config cfg;
    cfg.vocab_size = 2048;
    cfg.block_size = 128;
    cfg.n_layer    = 4;
    cfg.n_head     = 4;
    cfg.n_embd     = 128;

    GPT2 model(cfg);
    std::cout << "Model initialized with " << model.num_parameters << " parameters.\n";

    int B = 8; // Batch size
    DataLoader train_loader("train.bin", B, cfg.block_size);

    // Step 0 sanity check: loss should be ~ln(2048) = 7.62
    train_loader.random_batch();
    float init_loss = model.forward(train_loader.inputs.data(), train_loader.targets.data(), B, cfg.block_size);
    std::cout << "Step 0 Loss: " << init_loss << " (expected ~7.62)\n\n";

    // Training Loop
    int total_steps = 500;
    for (int step = 1; step <= total_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();

        train_loader.random_batch();
        model.zero_grad();
        float loss = model.forward(train_loader.inputs.data(), train_loader.targets.data(), B, cfg.block_size);
        model.backward(train_loader.inputs.data(), train_loader.targets.data(), B, cfg.block_size);
        model.update(1e-3f);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (step % 25 == 0 || step == 1) {
            std::cout << "Step " << step << "/" << total_steps
                      << " | Loss: " << loss
                      << " | Time: " << ms << " ms/step\n";
        }
    }

    model.save_checkpoint("gpt2_shakespeare.bin");

    // Generate sample Shakespeare after training!
    std::string prompt = "First Citizen:\n";
    auto prompt_ids = tokenizer.encode(prompt);
    auto generated_ids = model.generate(prompt_ids, 60, 0.8f);

    std::cout << "\n=== GENERATED SHAKESPEARE ===\n";
    std::cout << tokenizer.decode(generated_ids) << "\n";

    return 0;
}