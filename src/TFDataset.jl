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


