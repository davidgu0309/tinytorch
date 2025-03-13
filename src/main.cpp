#include "../include/computational_dag.hpp"
#include "../include/optimizer/gd.hpp"
#include "../include/tensor_operation/all.hpp"

#define TYPE double
#define INPUT_SHAPE {5}
#define OUTPUT_SHAPE {}
#define NUM_EPOCHS 5
#define NUM_DATA_POINTS 1
#define LEARNING_RATE 0.1

using namespace tinytorch;
using namespace tensor;

Addition<TYPE> addition;
Hadamard<TYPE> hadamard;
Matmul<TYPE> matmul_op;
Power<TYPE> power;
Lp<TYPE> lp_op;


Tensor<TYPE> testFunction(Tensor<TYPE> input) {
    return aggregate<TYPE, aggregator::sum<TYPE>>(input, 1);
}

Tensor<TYPE> generateX(size_t num_data_points) {
    Shape data_shape = INPUT_SHAPE;
    data_shape.insert(data_shape.begin(), num_data_points);
    return realUniform<TYPE>(data_shape, -100, 100);
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
    InputId x = model.addInput(INPUT_SHAPE);
    InputId y = model.addInput(OUTPUT_SHAPE);
    ParameterId w = model.addParameter(INPUT_SHAPE);

    ComputationalDAGNode<TYPE> dot_node(matmul_op, {{x, INPUT}, {w, PARAMETER}});
    graph::NodeId d_node = model.addNode(dot_node);  

    ComputationalDAGNode<TYPE> loss_node(lp_op, {{d_node, NODE}, {y, INPUT}});
    graph::NodeId l_node = model.addNode(loss_node);  

    model.getEntryPoint() = d_node;
    model.getExitPoint() = l_node;

    Tensor<TYPE> train_X = generateX(NUM_DATA_POINTS);
    Tensor<TYPE> train_y = testFunction(train_X);

    /*
    std::cout << "X" << std::endl;
    std::cout << train_X << std::endl;

    std::cout << "y" << std::endl;
    std::cout << train_y << std::endl;
    */

    // model.getInput(0) = ones<TYPE>({NUM_DATA_POINTS, 5});
    // model.getInput(1) = ones<TYPE>({NUM_DATA_POINTS});

    model.getInput(0) = ones<TYPE>(INPUT_SHAPE);
    model.getInput(1) = Tensor<TYPE>(0);

    model.getParameter(0) = iota<TYPE>(INPUT_SHAPE);

    model.forward();
    std::cout << model.get(d_node).result_ << "\n";
    std::cout << model.get(l_node).result_ << "\n";

    model.backward();
    std::cout << model.get(l_node).jacobi_[0] << "\n";
    std::cout << model.get(l_node).jacobi_[1] << "\n";

    std::cout << model.get(d_node).jacobi_[0] << "\n";
    std::cout << model.get(d_node).jacobi_[1] << "\n";


    // gradient_descent(model, train_X, train_y, LEARNING_RATE, NUM_EPOCHS);

    return 0;
}
