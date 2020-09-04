#include "libtorch_learning/dcgandiscriminatorimpl.h"
#include "libtorch_learning/dcgangeneratorimpl.h"

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    // The size of the noise vector fed to the generator.
    const int64_t kNoiseSize = 100;

    // The batch size for training.
    const int64_t kBatchSize = 64;

    // The number of epochs to train.
    const int64_t kNumberOfEpochs = 30;

    // Where to find the MNIST dataset.
    const char* kDataFolder = "/home/shreyas/.datasets/mnist/";

    // After how many batches to create a new checkpoint periodically.
    const int64_t kCheckpointEvery = 200;

    // How many images to sample at every checkpoint.
    const int64_t kNumberOfSamplesPerCheckpoint = 10;

    // Set to `true` to restore models and optimizers from previously saved
    // checkpoints.
    const bool kRestoreFromCheckpoint = false;

    // After how many batches to log a new update with the loss value.
    const int64_t kLogInterval = 10;

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    DCGANGenerator generator(kNoiseSize);
    generator->to(device);

    DCGANDiscriminator discriminator;
    discriminator->to(device);


   return 0;
}
