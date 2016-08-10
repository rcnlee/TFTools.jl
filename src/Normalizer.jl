#*****************************************************************************
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
import TensorFlow.API: shape, gather, concat, expand_dims, div_

type Normalizer
    ndim::Int64
    maxes::Array{Float32}
    mins::Array{Float32}
end

function Normalizer(data::TFDataset)
    N = ndims(data.X)
    rmdims = collect(1:N-1)
    #find max/mins per column over entire dataset
    maxes = maximum(data.X, rmdims)
    mins = minimum(data.X, rmdims)
    norm = Normalizer(N, maxes, mins)
    norm
end

#"""
#"""
function normalize01(norm::Normalizer, Xin::Tensor)
    N = norm.ndim
    maxes = constant(norm.maxes)
    mins = constant(norm.mins)
    tileshape = concat(Tensor(0), Tensor([gather(shape(Xin), Tensor(collect(0:N-2))); Tensor([1])]))
    tiledmaxes = tile(maxes, tileshape)
    tiledmins = tile(mins, tileshape)
    Xout = div_((Xin - tiledmins), (tiledmaxes - tiledmins))
    Xout
end

function normalize(norm::Normalizer, Xin::Tensor, minval::Float64=-1.0, 
    maxval::Float64=1.0)
    
    X01 = normalize(Xin)
    max = constant(maxval)
    min = constant(minval)
    shape = get_shape(Xin)
    tiledmax = tile(max, Tensor(shape))
    tiledmin = tile(min, Tensor(shape))
    Xout = X01 * (tiledmax - tiledmin) + tiledmin
    Xout
end

