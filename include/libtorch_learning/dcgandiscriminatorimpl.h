#ifndef DCGANDISCRIMINATORIMPL_H
#define DCGANDISCRIMINATORIMPL_H
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>


class DCGANDiscriminatorImpl : public torch::nn::Module
{
public:
    DCGANDiscriminatorImpl();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_, conv2_, conv3_, conv4_;
    torch::nn::BatchNorm2d batch_norm2_, batch_norm3_;
};
TORCH_MODULE(DCGANDiscriminator);

#endif // DCGANDISCRIMINATORIMPL_H
