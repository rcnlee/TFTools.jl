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

"""
TensorFlow components and tools
"""
module TFTools

export TFDataset, TFDatasets, next_batch, num_examples, Softmux, out, to_tensor, 
    getX, getY

using TensorFlow
import TensorFlow: DT_FLOAT32
import TensorFlow.API: relu, softmax, mul, cast
using Iterators

type TFDataset{Tx, Ty}
    X::Array{Tx,2}
    Y::Vector{Ty}
end

type TFDatasets
    train::TFDataset
    validation::TFDataset
    test::TFDataset
end

function TFDataset(DX, DY)
    X = convert(Array, DX)
    Y = convert(Array, DY)
    TFDataset(X, Y)
end

getX(ds::TFDataset) = Tensor(ds.X)
getY(ds::TFDataset) = Tensor(ds.Y)

function num_examples(ds::TFDataset)
    sizeX = size(ds.X)
    sizeY = size(ds.Y)
    @assert sizeX[1] == sizeY[1]
    sizeY[1]
end

function next_batch(ds::TFDataset, batch_size::Int64)
    start_index = 1 #blah
    end_index = 100 #blah
    x = ds.X[start_index:end_index, :]
    y = ds.Y[start_index:end_index] #label is dense (not one-hot)
    return (x, y)
end

"""
Soft multiplexer (selector) component that can be learned using gradient
descent
"""
type Softmux
    n_muxinput::Int64
    n_muxselect::Int64
    hidden_units::Vector{Int64}
    units::Vector{Int64}
    weights::Vector{Variable}
    biases::Vector{Variable}
    layers::Vector{Any}
    muxin::Tensor
    muxselect::Tensor
    nnout::Tensor
    muxout::Tensor
end

function Softmux(n_muxinput::Int64, n_muxselect::Int64, 
    hidden_units::Vector{Int64}, muxin::Tensor, muxselect::Tensor)

    @assert !isempty(hidden_units) 

    units = [n_muxselect; hidden_units; n_muxinput]
    n_layers = length(units) - 1
    weights = Array(Variable, n_layers)
    biases = Array(Variable, n_layers)    
    i = 1
    for (n1, n2) in partition(units, 2, 1)
        weights[i] = Variable(randn(Tensor, [n1, n2]))
        biases[i] = Variable(randn(Tensor, [n2]))
        i += 1
    end

    layers = Array(Any, n_layers)
    layers[1] = relu(muxselect * weights[1] + biases[1])
    for i = 2:n_layers-1
        layers[i] = relu(layers[i-1] * weights[i] + biases[i])
    end
    
    # last layer is softmax
    layers[end] = softmax(layers[end-1] * weights[end] + biases[end])
    nnout = layers[end] #softmax select over inputs
    # mux output is the soft selected input
    muxout = sum(mul(muxin, nnout)) #single output

    Softmux(n_muxinput, n_muxselect, hidden_units, units,
        weights, biases, layers, muxin, muxselect, nnout, muxout)
end

function out(mux::Softmux)
    mux.muxout
end

function hardout(mux::Softmux, muxin::Placeholder, muxselect::Placeholder)
    #eval...
    mux.nnout
end

end # module
