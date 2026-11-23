#include "tokeniser.hpp"
#include "gpt2.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

int main() {
    // 1. Load your saved 2048-token BPE tokenizer
    BPETokenizer tokenizer;
    tokenizer.load("tokenizer_2048.bin");

    // 2. Initialize model with the EXACT same config used during training
    GPT2Config cfg;
    cfg.vocab_size = 2048;
    cfg.block_size = 128;
    cfg.n_layer    = 4;
    cfg.n_head     = 4;
    cfg.n_embd     = 128;

    GPT2 model(cfg);
    model.load_checkpoint("gpt2_shakespeare.bin");

    // 3. Interactive Generation Loop
    std::string prompt;
    int max_tokens = 80;
    float temperature = 0.8f;

    std::cout << "\n=== GPT2-CPP SHAKESPEARE INFERENCE ===\n";
    std::cout << "Type a prompt (or 'quit' to exit). Tip: use \\n for newlines (e.g., ROMEO:\\n)\n";

    while (true) {
        std::cout << "\nPrompt > ";
        if (!std::getline(std::cin, prompt) || prompt == "quit") {
            break;
        }
        if (prompt.empty()) {
            prompt = "First Citizen:\n";
        }

        // Convert literal "\n" typed in terminal into actual newline characters
        size_t pos = 0;
        while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
            prompt.replace(pos, 2, "\n");
            pos += 1;
        }

        std::vector<int> prompt_ids = tokenizer.encode(prompt);
        if (prompt_ids.empty()) {
            std::cout << "Empty prompt tokens, try again.\n";
            continue;
        }

        // Use a fresh random seed per prompt so outputs vary naturally
        uint32_t seed = static_cast<uint32_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<int> output_ids = model.generate(prompt_ids, max_tokens, temperature, seed);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int generated_count = static_cast<int>(output_ids.size() - prompt_ids.size());

        std::cout << "\n--- Output (" << generated_count << " tokens in " << ms << " ms) ---\n";
        std::cout << tokenizer.decode(output_ids) << "\n";
        std::cout << "-----------------------------------------\n";
    }

    return 0;
}
