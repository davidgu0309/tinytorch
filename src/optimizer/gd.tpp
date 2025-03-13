namespace tinytorch{

    template<typename T>
    void gradient_descent_iteration(ComputationalDAG<T>& model, const Tensor<T>& data, const Tensor<T>& targets, const T learning_rate){
        model.getInput(0) = data;
        model.getInput(1) = targets;
        model.forward();  
        model.backward();
        std::cout << "Backward completed \n";

        std::cout << "Loss: " << aggregate<T, aggregator::mean<T>>(model.get(model.getExitPoint()).result_, 0) << "\n";

        std::vector<graph::NodeId> topo_order = model.topoOrder();
        for(graph::NodeId node_id : topo_order){
            for(size_t i = 0; i < model.get(node_id).operand_descriptor_.size(); ++i){
                OperandDescriptor& operand_descriptor = model.get(node_id).operand_descriptor_[i];
                if(operand_descriptor.operand_type_ == PARAMETER){
                    ParameterId operand_parameter_id = operand_descriptor.id_.parameter_id_;
                    Tensor<T> gradient = model.get(node_id).jacobi_[i]; 
                    std::cout << "Gradient: \n" << gradient << std::endl;
                    gradient = aggregate<T, aggregator::mean<T>>(gradient, gradient.shape().size() - 1);    // TODO maybe fix memory leak
                    model.getParameter(operand_parameter_id) = add<T>(neg<T>(mul<T>(constant(gradient.shape(), learning_rate), gradient)), 
                                                                    model.getParameter(operand_parameter_id));  //TODO replace with scalar learning_rate 
                    std::cout << "Model parameters: " << model.getParameter(operand_parameter_id) << "\n";
                }
            }
        }

        
        
    }

    template <typename T>
    T learning_rate_schedule(const T max_learning_rate, const size_t current_epoch, const size_t epochs) {
        // quick and dirty linear learning rate scheduler without warmup
        return max_learning_rate * (epochs-current_epoch)/epochs;   // assuming current_epoch ranging from 0 to epochs-1
    }

    template<typename T>
    void gradient_descent(ComputationalDAG<T>& model, const Tensor<T>& data, const Tensor<T>& targets, const T learning_rate, const size_t epochs){
        for (int i=0; i<epochs; i++) {
            T current_learning_rate = learning_rate_schedule<T>(learning_rate, i, epochs);
            gradient_descent_iteration<T>(model, data, targets, current_learning_rate);
        }
    }

} // namespace tinytorch