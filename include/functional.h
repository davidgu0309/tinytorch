#include <tensor.h>

namespace tinytorch {
    Tensor add(const Tensor& a, Tensor& b);

    Tensor mul(const Tensor& a, Tensor& b);

    Tensor matmul(const Tensor& a, Tensor& b);

    Tensor relu(const Tensor& a);

    Tensor sigmoid(const Tensor& a);

    Tensor softmax(const Tensor& a);

    Tensor cross_entropy(const Tensor& logits, const Tensor& target);
}