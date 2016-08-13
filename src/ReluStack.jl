# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter Simulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter Simulator (RLES) includes the following
# third party software. The SISLES.jl package is licensed under the MIT Expat
# License: Copyright (c) 2014: Youngjun Kim.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *****************************************************************************

using TensorFlow
import TensorFlow.API: relu

"""
Multiple relu layers
"""
type ReluStack
   n_inputs::Int64
   n_units::Vector{Int64}
   weights::Vector{Variable}
   biases::Vector{Variable}
   layers::Vector{Tensor}
   out::Tensor
end

function ReluStack(input::Tensor, n_units::Vector{Int64})
    n_inputs = get_shape(input)[end]

    #if no units then just return input
    if isempty(n_units)
        return ReluStack(n_inputs, Int64[], Variable[], Variable[], Tensor[], input)
    end
    
    n_layers = length(n_units)
    weights = Array(Variable, n_layers)
    biases = Array(Variable, n_layers)    
    layers = Array(Tensor, n_layers)
    
    n1 = n_units[1]
    weights[1] = Variable(randn(Tensor, [n_inputs, n1]))
    biases[1] = Variable(randn(Tensor, [n1]))
    layers[1] = relu(input * weights[1] + biases[1])

    for i = 2:n_layers
        n0 = n_units[i-1]
        n1 = n_units[i]
        weights[i] = Variable(randn(Tensor, [n0, n1]))
        biases[i] = Variable(randn(Tensor, [n1]))
        layers[i] = relu(layers[i-1] * weights[i] + biases[i])
    end
    out = layers[end]

    ReluStack(n_inputs, n_units, weights, biases, layers, out)
end

function out(relustack::ReluStack)
    relustack.out
end

