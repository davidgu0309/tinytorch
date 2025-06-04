#include "../include/computational_dag.hpp"
#include "../include/data_pipeline/data_loader.hpp"
#include "../include/optimizer/gd.hpp"
#include "../include/tensor_operation/all.hpp"

#define TYPE double

#define INPUT_SIZE 5
#define HL1_SIZE 10
#define OUTPUT_SIZE 5

#define NUM_EPOCHS 30
#define NUM_DATA_POINTS 300
#define LEARNING_RATE 5.5e-4

using namespace tinytorch;
using namespace tensor;

Addition<TYPE> addition;
Hadamard<TYPE> hadamard;
Matmul<TYPE> matmul_op;
Power<TYPE> power;
Lp<TYPE> lp_op;
ReLU<TYPE> relu_op;

template <typename T>
struct Cube{

    Cube(){}

    T operator()(const std::vector<T> operands) const{
        assert(operands.size() == 1);
        return operands[0] * operands[0] * operands[0];
    }

    T backward(const size_t input_idx, const std::vector<T> operands) const{
        assert(input_idx == 0);
        return 3 * operands[0] * operands[0];
    }

};

template <typename T>
struct TensorCube : ComponentwiseOperation<T, Cube<T>> {

    TensorCube(){}

    using ComponentwiseOperation<T, Cube<T>>::operator ();

    using ComponentwiseOperation<T, Cube<T>>::backward;

};

TensorCube<TYPE> tc;

template <typename T>
struct TensorMean : TensorOperation<T> {

    Tensor<T> operator()(const std::vector<Tensor<T>> operands) const {
        assert(operands.size() == 1);
        T s = 0;
        for(const T entry : operands[0].data()) s += entry;
        return Tensor<T>(s / operands[0].data().size());
    }

    Tensor<T> backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const {
        assert(input_idx == 0);
        return constant<T>(operands[0].shape(), 1.0 / operands[0].data().size());
    }
};

TensorMean<TYPE> mean;

Tensor<TYPE> testFunction(Tensor<TYPE> input) {
    return tc({input});
}

Tensor<TYPE> generateX(size_t num_data_points) {
    Shape data_shape = {INPUT_SIZE};
    data_shape.insert(data_shape.begin(), num_data_points);
    return realUniform<TYPE>(data_shape, -1, 1);
}

// Tensor<TYPE> generateY(Tensor<TYPE> input_data) {
    // Shape data_shape = OUTPUT_SHAPE;
    // size_t num_data_points = input_data.shape()[0];
    // data_shape.insert(data_shape.begin(), num_data_points);
    // Tensor<TYPE> y(data_shape);

    // for (size_t i=0; i<num_data_points; i++) {
    //     y
    // }
// }

int main() {

    ComputationalDAG<TYPE> model;

    InputId x = model.addInput({INPUT_SIZE});
    InputId y = model.addInput({OUTPUT_SIZE});

    ParameterId l1w = model.addParameter({INPUT_SIZE, HL1_SIZE});
    ParameterId l1b = model.addParameter({HL1_SIZE});
    ParameterId l2w = model.addParameter({HL1_SIZE, OUTPUT_SIZE});
    ParameterId l2b = model.addParameter({OUTPUT_SIZE});

    // LAYER 1
    ComputationalDAGNode<TYPE> L1W_node(matmul_op, {{x, INPUT}, {l1w, PARAMETER}});
    graph::NodeId l1w_node = model.addNode(L1W_node);

    ComputationalDAGNode<TYPE> L1B_node(addition, {{l1w_node, NODE}, {l1b, PARAMETER}});
    graph::NodeId l1b_node = model.addNode(L1B_node);  

    ComputationalDAGNode<TYPE> L1A_node(relu_op, {{l1b_node, NODE}});
    graph::NodeId l1a_node = model.addNode(L1A_node);  

    // LAYER 2
    ComputationalDAGNode<TYPE> L2W_node(matmul_op, {{l1a_node, NODE}, {l2w, PARAMETER}});
    graph::NodeId l2w_node = model.addNode(L2W_node);

    ComputationalDAGNode<TYPE> L2B_node(addition, {{l2w_node, NODE}, {l2b, PARAMETER}});
    graph::NodeId l2b_node = model.addNode(L2B_node);  

    ComputationalDAGNode<TYPE> L2A_node(relu_op, {{l2b_node, NODE}});
    graph::NodeId l2a_node = model.addNode(L2A_node);  

    // LOSS
    ComputationalDAGNode<TYPE> loss_node(lp_op, {{l2a_node, NODE}, {y, INPUT}});
    graph::NodeId l_node = model.addNode(loss_node);  

    ComputationalDAGNode<TYPE> mean_node(mean, {{l_node, NODE}});
    graph::NodeId m_node = model.addNode(mean_node);

    model.getEntryPoint() = l1w_node;
    model.getExitPoint() = m_node;

    Tensor<TYPE> train_X = generateX(NUM_DATA_POINTS);
    Tensor<TYPE> train_y = testFunction(train_X);

    /*
    std::cout << "X" << std::endl;
    std::cout << train_X << std::endl;

    std::cout << "y" << std::endl;
    std::cout << train_y << std::endl;
    */

    // model.getInput(0) = ones<TYPE>({NUM_DATA_POINTS, 5});
    // model.getInput(1) = zeros<TYPE>({NUM_DATA_POINTS});

    model.getInput(0) = ones<TYPE>({INPUT_SIZE});
    model.getInput(1) = Tensor<TYPE>({OUTPUT_SIZE}, {0, 0, 0, 0, 0});

    model.getParameter(l1w) = iota<TYPE>({INPUT_SIZE, HL1_SIZE});
    model.getParameter(l1b) = zeros<TYPE>({HL1_SIZE});

    model.getParameter(l2w) = iota<TYPE>({HL1_SIZE, OUTPUT_SIZE});
    model.getParameter(l2b) = zeros<TYPE>({OUTPUT_SIZE});

    model.forward();
    std::cout << "\n----------------- Forward Test -------------------\n";
    std::cout << model.get(l1w_node).result_ << "\n";
    std::cout << model.get(l1b_node).result_ << "\n";
    std::cout << model.get(l1a_node).result_ << "\n";
    std::cout << model.get(l2w_node).result_ << "\n";
    std::cout << model.get(l2b_node).result_ << "\n";
    std::cout << model.get(l2a_node).result_ << "\n";
    std::cout << model.get(l_node).result_ << "\n";
    std::cout << model.get(m_node).result_ << "\n";

    model.backward();
    std::cout << "\n----------------- Backward Test -------------------\n";
    std::cout << model.get(l_node).jacobi_[0] << "\n";
    std::cout << model.get(l_node).jacobi_[1] << "\n";
    std::cout << model.get(l1w_node).jacobi_[0] << "\n";
    std::cout << model.get(l1w_node).jacobi_[1] << "\n";

    gradient_descent(model, train_X.unstack(0), train_y.unstack(0), LEARNING_RATE, NUM_EPOCHS);

    std::vector<Tensor<int>> data = load_csv<int>("data/preprocessed_data.csv", 2);

    for(auto t : data) std::cout << t << std::endl;

    return 0;
}
