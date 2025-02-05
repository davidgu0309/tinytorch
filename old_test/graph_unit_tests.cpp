#include "../graph/graph.hpp"

#include <iostream>

namespace tinytorch {

void graphUnitTests(){
    // TO DO: improve framework and rewrite this
    graph::Graph<int> graph;
    for(int i = 1; i <= 5; ++i) graph.addNode(i);
    std::cout << "Size test 1: " << (graph.size() == 5 ? "Passed" : "Failed") << std::endl;
    std::cout << std::endl;
}   
    
} // namespace tinytorch
