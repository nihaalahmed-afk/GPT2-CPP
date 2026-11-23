#include "tokeniser.hpp"
#include "dataloader.hpp"

int main() {
    BPETokenizer tokenizer;
    tokenizer.load("tokenizer_2048.bin");

    DataLoader train_loader("train.bin", 4, 64);
    train_loader.random_batch();

    for (int t = 0; t < 8; ++t) {
        int x = train_loader.inputs[t];
        int y = train_loader.targets[t];
        std::cout << "Step " << t << ": [" << escape_bytes(tokenizer.decode({x}))
                  << "] (" << x << ")  --->  [" << escape_bytes(tokenizer.decode({y}))
                  << "] (" << y << ")\n";
    }

    return 0;
}
