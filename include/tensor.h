#include <vector>
#include <memory>

namespace tinytorch {

class Tensor {
    public:
        Tensor();
        Tensor(const std::vector<size_t>& shape);
        Tensor(const std::vector<float>& data,
                const std::vector<size_t> shape);

        const std::vector<size_t>& shape() const {return shape_;}
        size_t size() const;
        float* data() {return data_.get();}
        const float* data() const {return data_.get();}

    private:
        std::unique_ptr<float[]> data_;
        std::vector<size_t> shape_;
    };
}

