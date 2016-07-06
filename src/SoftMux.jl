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
import TensorFlow.API: mul, softmax, arg_max, cast, reduce_sum,
    expand_dims, tile

"""
Soft multiplexer (selector) component that can be learned using gradient
descent
"""
type SoftMux
    n_muxinput::Int64
    hidden_units::Vector{Int64}
    muxin::Tensor
    muxselect::Tensor
    nn::ReluStack
    weight::Variable
    bias::Variable
    nnout::Tensor
    muxout::Tensor
    hardselect::Tensor
    hardout::Tensor
end

function SoftMux(n_muxinput::Int64, 
    hidden_units::Vector{Int64}, 
    muxin::Tensor, 
    muxselect::Tensor)

    @assert !isempty(hidden_units) 

    relustack = ReluStack(muxselect, hidden_units)
    reluout = out(relustack)
    
    # last layer is softmax
    # softmax select over inputs
    n0 = get_shape(reluout)[end]
    n1 = n_muxinput 
    weight = Variable(randn(Tensor, [n0, n1]))
    bias = Variable(randn(Tensor, [n1]))
    nnout = softmax(reluout * weight + bias)
    
    # mux output is the soft selected input
    hardselect = arg_max(nnout, Tensor(1)) #hardened dense selected channel, 0-indexed
    hardselect_1h = one_hot(hardselect, Tensor(n_muxinput))
    muxin_rank = ndims(muxin)
    if muxin_rank == 2 #TODO: avoid switching if possible
        muxout = reduce_sum(mul(muxin, nnout), Tensor(1))
        hardout = reduce_sum(mul(muxin, hardselect_1h), Tensor(1))
    elseif muxin_rank == 3
        muxout = reduce_sum(mul3(muxin, nnout), Tensor(2))
        hardout = reduce_sum(mul3(muxin, hardselect_1h), Tensor(2))
    else
        error("Not supported! (rank=$(muxin_rank))")
    end

    SoftMux(n_muxinput, hidden_units, muxin, muxselect, relustack, weight, bias,
        nnout, muxout, hardselect, hardout)
end

function mul3(X::Tensor, y::Tensor)
    n_time = get_shape(X)[2]
    tmp = expand_dims(y, Tensor(1))
    Y = tile(tmp, Tensor(Int32[1, n_time, 1]))
    out = mul(X, Y) #element-wise mul
    out
end

function out(mux::SoftMux)
    mux.muxout
end

function hardselect(mux::SoftMux)
    mux.hardselect
end

function hardout(mux::SoftMux)
    mux.hardout
end

function hardselect(mux::SoftMux)
    mux.hardselect
end


