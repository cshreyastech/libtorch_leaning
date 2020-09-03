#include "libtorch_learning/dcgangeneratorimpl.h"

DCGANGeneratorImpl::DCGANGeneratorImpl(int kNoiseSize)
    : conv1_(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                    .bias(false)),
            batch_norm1_(256),
            conv2_(torch::nn::ConvTranspose2dOptions(256, 128, 3)
                   .stride(2)
                   .padding(1)
                   .bias(false)),
            batch_norm2_(128),
            conv3_(torch::nn::ConvTranspose2dOptions(128, 64, 4)
                   .stride(2)
                   .padding(1)
                   .bias(false)),
            batch_norm3_(64),
            conv4_(torch::nn::ConvTranspose2dOptions(64, 1, 4)
                   .stride(2)
                   .padding(1)
                   .bias(false)) {

    register_module("conv1", conv1_);
    register_module("batch_norm1", batch_norm1_);

    register_module("conv2", conv2_);
    register_module("batch_norm2", batch_norm2_);

    register_module("conv3", conv3_);
    register_module("batch_norm3", batch_norm3_);

    register_module("conv4", conv4_);
}

torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x) {
    x = torch::relu(batch_norm1_(conv1_(x)));
    x = torch::relu(batch_norm2_(conv2_(x)));
    x = torch::relu(batch_norm3_(conv3_(x)));
    x = torch::tanh(conv4_(x));

    return x;
}
