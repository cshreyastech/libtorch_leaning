#ifndef DCGANGENERATORIMPL_H
#define DCGANGENERATORIMPL_H

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>

class DCGANGeneratorImpl : public torch::nn::Module
{
public:
    DCGANGeneratorImpl(int kNoiseSize);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::ConvTranspose2d conv1_, conv2_, conv3_, conv4_;
    torch::nn::BatchNorm2d batch_norm1_, batch_norm2_, batch_norm3_;


};

TORCH_MODULE(DCGANGenerator);
//DCGANGenerator generator(kNoiseSize);

#endif // DCGANGENERATORIMPL_H
