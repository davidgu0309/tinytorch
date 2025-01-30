/**
 * @file autograd.hpp
 * 
 * @brief Do we even want this file?
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

// [f(g(h(x)))]' = f'(g(h(x))) * g'(h(x)) * h'(x)
// think about equivalent to pytorch model.eval() model.train()
// sigmoid'(x) = sigmoid(x) * (1-sigmoid(x))

// Jacobi-matrix: the matrix that collects the derivatives for R^n -> R^m in a mxn matrix. 
// g'(h(x)) * h'(x)
// that means that if h(x) is R^n and g(h(x)) is R^m, the shapes work out because (g(h(x)))' is m-dimensional 

// d(x*W+b)/dW_{11} d(x*W+b)/dW_{12} d(x*W+b)/dW_{13}
// d(x*W+b)/dW_{21} _ _

// W is (dim_in, dim_out), x is dim_in, so x*W+b is dim_out, so the Jacobian should be (dim_out, dim_in, dim_in)

// y = W*x
// (---dy1_dW----)
// (---dy2_dW----)

// for y = x*W this above should be transposed! The first column is (---dy1_dW----)^T, the second (---dy2_dW----)^T and so on

// multi-dimensional: dloss(f(x*W+b), y)/dW = loss'(f(x*W+b),y) * f'(x*W+b) *       (x*W+b)'
//                                                  vector      diag(x > 0 ? 1:0)       
//                                                  R^dim_out   R^{dim_out x dim_out}  R^{dim_out,dim_out,dim_in}   

// all of the above is correct for the convention that we multiply matrices from the left, however since we multiply from the right, we do
// [f(g(h(x)))]' = h'(x) * g'(h(x)) * f'(g(h(x))) and we need to reverse the order of all terms above

// f(x) = relu(x) -> f'(x) = x > 0 ? 1:0                            
// d(x*W+b)/dW_{ij} is a one-hot vector with j-entry = x_j 


