#include <random>

namespace tinytorch {
    std::random_device rd;
    std::mt19937 gen(rd());
}


// namespace tinytorch {
//     template <typename T>
//     T uniform
// }