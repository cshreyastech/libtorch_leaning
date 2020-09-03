#include "libtorch_learning/dcgandiscriminatorimpl.h"

DCGANDiscriminatorImpl::DCGANDiscriminatorImpl()
    : conv1_(torch::nn::Conv2dOptions(1, 64, 4)
             .stride(2)
             .padding(1)
             .bias(false)),
      conv2_(torch::nn::Conv2dOptions(64, 128, 4)
              .stride(2)
              .padding(1)
              .bias(false)),
     batch_norm2_(128),
     conv3_(torch::nn::Conv2dOptions(128, 256, 4)
             .stride(2)
             .padding(1)
             .bias(false)),
    batch_norm3_(256),
    conv4_(torch::nn::Conv2dOptions(256, 1, 3)
            .stride(2)
            .padding(0)
            .bias(false))
{
    register_module("conv1", conv1_);

    register_module("conv2", conv2_);
    register_module("batch_norm2", batch_norm2_);

    register_module("conv3", conv3_);
    register_module("batch_norm3", batch_norm3_);

    register_module("conv4", conv4_);
}

torch::Tensor DCGANDiscriminatorImpl::forward(torch::Tensor x) {

    x = torch::relu(conv1_(x));
    x = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))(x);

    x = torch::relu(batch_norm2_(conv2_(x)));
    x = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))(x);

    x = torch::relu(batch_norm3_(conv3_(x)));
    x = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))(x);

    x = torch::nn::Sigmoid()(x);

    return x;
}
