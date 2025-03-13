namespace tinytorch{

    template<typename T>
    void gradient_descent_iteration(ComputationalDAG<T>& model, const std::vector<Tensor<T>>& data, const std::vector<Tensor<T>>& targets, const T learning_rate){
        std::vector<graph::NodeId> topo_order = model.topoOrder();  //TODO any list of nodes is enough
        for (graph::NodeId node_id : topo_order) {
            std::vector<Tensor<T>>& gradient_accumulators = model.get(node_id).jacobi_accumulator_;
            for (Tensor<T>& accumulator : gradient_accumulators) {
                accumulator.clear();
            }
        }

        Tensor<T> loss = 0;
        size_t num_data_points = data.size();
        for (size_t i=0; i<num_data_points; i++) {
            model.getInput(0) = data[i];
            model.getInput(1) = targets[i];
            model.forward();  
            loss = add<T>(loss, model.get(model.getExitPoint()).result_);
            model.backward();

            for (graph::NodeId node_id : topo_order) {
                std::vector<Tensor<T>>& gradient_accumulators = model.get(node_id).jacobi_accumulator_;
                size_t i=0;
                for (Tensor<T>& accumulator : gradient_accumulators) {
                    accumulator = add<T>(accumulator, model.get(node_id).jacobi_[i++]);   //not necessary for non-parameters
                }
            }
        }

        std::cout << "Loss: " << mul<T>(loss, constant<T>(loss.shape(), (T) 1/num_data_points)) << "\n";

        for(graph::NodeId node_id : topo_order){
            for(size_t i = 0; i < model.get(node_id).operand_descriptor_.size(); ++i){
                OperandDescriptor& operand_descriptor = model.get(node_id).operand_descriptor_[i];
                if(operand_descriptor.operand_type_ == PARAMETER){
                    ParameterId operand_parameter_id = operand_descriptor.id_.parameter_id_;
                    Tensor<T> gradient = model.get(node_id).jacobi_accumulator_[i]; 
                    std::cout << "Gradient: \n" << gradient << std::endl;
                    //gradient = aggregate<T, aggregator::mean<T>>(gradient, gradient.shape().size() - 1);    // TODO maybe fix memory leak
                    model.getParameter(operand_parameter_id) = add<T>(neg<T>(mul<T>(constant(gradient.shape(), learning_rate/num_data_points), gradient)), 
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
    void gradient_descent(ComputationalDAG<T>& model, const std::vector<Tensor<T>>& data,  
                             const std::vector<Tensor<T>>& targets, const T learning_rate, const size_t epochs)
    {
        for (int i=0; i<epochs; i++) {
            T current_learning_rate = learning_rate_schedule<T>(learning_rate, i, epochs);
            gradient_descent_iteration<T>(model, data, targets, current_learning_rate);
        }
    }

} // namespace tinytorch