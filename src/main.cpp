#include "../include/tensor.hpp"
#include "../include/layer.hpp"

constexpr size_t EXAMPLE_SHAPE = 5;
#define EXAMPLE_TYPE double

int main() {
    tinytorch::Linear<EXAMPLE_TYPE> linear_layer(EXAMPLE_SHAPE, EXAMPLE_SHAPE);
    auto input = tinytorch::real_uniform<EXAMPLE_TYPE>({EXAMPLE_SHAPE}, 0, 1);
    std::cout << "Weights of linear layer: \n";
    std::cout << linear_layer.weights() << "\n\n";
    std::cout << "Bias of linear layer: \n";
    std::cout << linear_layer.bias() << "\n\n";
    std::cout << "Input: " << input << "\n";
    std::cout << "Output: " << linear_layer.forward(input);
    return 0;
}