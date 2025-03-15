
namespace tinytorch{

    template<typename T>
    std::vector<Tensor<T>> load_csv(std::string path, size_t batch_size){

        std::cout << path << std::endl;

        std::ifstream file(path);
        if(!file.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            return {};
        }

        std::vector<Tensor<T>> data;
        std::vector<std::vector<T>> batch;
        std::string line;

        while (std::getline(file, line)) {
            
            std::stringstream ss(line);
            
            std::vector<T> row;
            T value;

            char comma;

            while (ss >> value) {
                row.push_back(value);
                if (!(ss >> comma)) break;
            }

            if(batch.size() && row.size() != batch.back().size()){
                std::cerr << "All rows must contain the same number of entries!" << std::endl;
                return {};
            }

            batch.push_back(row);

            if(batch.size() == batch_size){
                data.push_back(Tensor<T>({batch.size(), batch[0].size()}));
                size_t k = 0;
                for(size_t i = 0; i < batch.size(); ++i){
                    for(size_t j = 0; j < batch[i].size(); ++j){
                        data.back().data()[k++] = batch[i][j];
                    }
                }
                batch.clear();
            }
        }

        if(batch.size()){
            data.push_back(Tensor<T>({batch.size(), batch[0].size()}));
            size_t k = 0;
            for(size_t i = 0; i < batch.size(); ++i){
                for(size_t j = 0; j < batch[i].size(); ++j){
                    data.back().data()[k++] = batch[i][j];
                }
            }
        }

        file.close();

        return data;
    }
}