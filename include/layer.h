#include <tensor.h>
#include <vector>
#include <memory>
#include <functional.h>

namespace tinytorch{

class Linear {
    public:
        Linear(size_t dim_in, size_t dim_out);
        
        Tensor forward(Tensor& input);

        size_t num_params() const {return dim_in * dim_out;}
        float* weights() {return weights_.get();}
        const float* bias() const {return bias_.get();}

    private:
        size_t dim_in, dim_out;
        std::unique_ptr<float> weights_;
        std::unique_ptr<float> bias_;

};

class ReLU {
    public:
        Tensor forward(Tensor& input);
};

class Conv1D {
    public:
        Tensor forward(Tensor& input);

    // TODO -----------complete -----------
};

}