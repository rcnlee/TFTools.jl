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

export TFDataset, TFDatasets, next_batch, num_examples, Softmux, out, hardout,
    getX, getY, OpsBlock, num_ops, getindex1, get_shape, hardselect

using TensorFlow
import TensorFlow: DT_FLOAT32 
import TensorFlow.API: relu, softmax, mul, cast, reduce_sum, pack, transpose_, 
    arg_max, expand_dims, tile
using Iterators

include(joinpath(dirname(@__FILE__), "TFSupplemental.jl"))

type TFDataset{Tx,Ty,N}
    X::Array{Tx,N}
    Y::Vector{Ty}
    num_examples::Int64
    index_in_epoch::Int64
    epochs_completed::Int64
end

type TFDatasets
    train::TFDataset
    validation::TFDataset
    test::TFDataset
end

function TFDataset(DX, DY)
    X = convert(Array, DX)
    Y = convert(Array, DY)

    sizeX = size(X)
    sizeY = size(Y)
    @assert sizeX[1] == sizeY[1]
    num_examples = sizeY[1]

    index_in_epoch = 0
    epochs_completed = 0
    TFDataset(X, Y, num_examples, index_in_epoch, epochs_completed)
end

getX(ds::TFDataset) = Tensor(ds.X)
getY(ds::TFDataset) = Tensor(ds.Y)
epochs_completed(ds::TFDataset) = ds.epochs_completed
num_examples(ds::TFDataset) = ds.num_examples
index_in_epoch(ds::TFDataset) = ds.index_in_epoch

function next_batch(ds::TFDataset, batch_size::Int64)
    start_index = ds.index_in_epoch + 1
    ds.index_in_epoch += batch_size
    if ds.index_in_epoch > ds.num_examples
        # Finished epoch
        ds.epochs_completed += 1
        # Shuffle the data
        perm = randperm(ds.num_examples)
        ds.X = getindex1(ds.X, perm)
        ds.Y = ds.Y[perm]
        start_index = 1
        ds.index_in_epoch = batch_size
        @assert ds.index_in_epoch <= ds.num_examples
    end
    end_index = ds.index_in_epoch
    x = getindex1(ds.X, start_index:end_index)
    y = ds.Y[start_index:end_index] #label is dense (not one-hot)
    return (x, y)
end

"""
getindex where the index for the first dim is given, others are ':'
this function exists to support varying dims in arrays
"""
function getindex1{T,N}(X::Array{T,N}, i1)
    if N == 2
        return X[i1, :]
    elseif N == 3
        return X[i1, :, :]
    elseif N == 1
        return X[i1]
    else
        error("Not implemented")
    end
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
    hardselect::Tensor
    hardout::Tensor
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
    # softmax select over inputs
    layers[end] = softmax(layers[end-1] * weights[end] + biases[end])
    nnout = layers[end] 
    
    # mux output is the soft selected input
    hardselect = arg_max(nnout, Tensor(1)) #hardened dense selected channel, 0-indexed
    hardselect_1h = one_hot(hardselect, Tensor(n_muxinput))
    muxin_rank = length(get_shape(muxin))
    if muxin_rank == 2
        muxout = reduce_sum(mul(muxin, nnout), Tensor(1))
        hardout = reduce_sum(mul(muxin, hardselect_1h), Tensor(1))
    elseif muxin_rank == 3
        muxout = reduce_sum(mul3(muxin, nnout), Tensor(2))
        hardout = reduce_sum(mul3(muxin, hardselect_1h), Tensor(2))
    else
        error("Not supported! (rank=$(muxin_rank))")
    end

    Softmux(n_muxinput, n_muxselect, hidden_units, units,
        weights, biases, layers, muxin, muxselect, nnout, muxout, hardselect, hardout)
end

function mul3(X::Tensor, y::Tensor)
    n_time = get_shape(X)[2]
    tmp = expand_dims(y, Tensor(1))
    Y = tile(tmp, Tensor(Int32[1, n_time, 1]))
    out = mul(X, Y) #element-wise mul
    out
end

# Is this stable/robust?
function get_shape(X::Tensor)
    dims = (map(d -> d[:value], X.x[:get_shape]())...)
    dims
end

function out(mux::Softmux)
    mux.muxout
end

function hardselect(mux::Softmux)
    mux.hardselect
end

function hardout(mux::Softmux)
    mux.hardout
end

function hardselect(mux::Softmux)
    mux.hardselect
end

type OpsBlock
    inputs::Tuple{Vararg{Tensor}}
    op_list::Vector{Function}
    outs::Vector{Tensor}
    output::Tensor
end

function OpsBlock(inputs::Tuple{Vararg{Tensor}}, op_list::Vector{Function})
    outs = map(f -> transpose_(f(inputs...)), op_list) #FIXME: remove the double transpose, but this is due to pack packing to the wrong dimension
    output = transpose_(pack(Tensor(outs)))

    OpsBlock(inputs, op_list, outs, output) 
end

function num_ops(ops::OpsBlock)
    length(ops.op_list)
end

function out(ops::OpsBlock)
    ops.output
end

end # module
