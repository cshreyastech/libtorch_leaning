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
    const int64_t kNumberOfEpochs = 2;

    // Where to find the MNIST dataset.
    const char* kDataFolder = "/home/shreyas/.datasets/mnist/";
//    const char* kDataFolder = "/home/shreyas/Downloads/mnist_dataset/";

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
//    torch::Device device(cuda_available ? torch::kCPU : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    DCGANGenerator generator(kNoiseSize);
    generator->to(device);

    DCGANDiscriminator discriminator;
    discriminator->to(device);

    // Assume the MNIST dataset is available under `kDataFolder`;
     auto dataset = torch::data::datasets::MNIST(kDataFolder)
                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                        .map(torch::data::transforms::Stack<>());
     const int64_t batches_per_epoch =
         std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

     auto data_loader = torch::data::make_data_loader(
         std::move(dataset),
         torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

     torch::optim::Adam generator_optimizer(
         generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
     torch::optim::Adam discriminator_optimizer(
         discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));

     if (kRestoreFromCheckpoint) {
       torch::load(generator, "generator-checkpoint.pt");
       torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
       torch::load(discriminator, "discriminator-checkpoint.pt");
       torch::load(
           discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
     }
     int64_t checkpoint_counter = 1;

     for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
       int64_t batch_index = 0;
       for (torch::data::Example<>& batch : *data_loader) {
         // Train discriminator with real images.
         discriminator->zero_grad();
         torch::Tensor real_images = batch.data.to(device);
         torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0).to(device);
         torch::Tensor real_output = discriminator->forward(real_images).to(device);
//         std::cout << real_labels.size(0) << std::endl;
//         std::cout << real_output.size(0) << ", " << real_output.size(1) << ", " << real_output.size(2) << ", " << std::endl;
//         std::cout << "---------" << std::endl;
         torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output[0], real_labels[0]);
         d_loss_real.backward();

         // Train discriminator with fake images.
         torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}).to(device);
         torch::Tensor fake_images = generator->forward(noise).to(device);
         torch::Tensor fake_labels = torch::zeros(batch.data.size(0)).to(device);
         torch::Tensor fake_output = discriminator->forward(fake_images.detach()).to(device);
         torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output[0], fake_labels[0]).to(device);
         d_loss_fake.backward();

         torch::Tensor d_loss = d_loss_real + d_loss_fake.to(device);
         discriminator_optimizer.step();

         // Train generator.
         generator->zero_grad();
         fake_labels.fill_(1);
         fake_output = discriminator->forward(fake_images);
         torch::Tensor g_loss = torch::binary_cross_entropy(fake_output[0], fake_labels[0]).to(device);
         g_loss.backward();
         generator_optimizer.step();

         std::printf(
             "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
             epoch,
             kNumberOfEpochs,
             ++batch_index,
             batches_per_epoch,
             d_loss.item<float>(),
             g_loss.item<float>());
       }
     }

     std::cout << "Training complete!" << std::endl;
     return 0;
}
