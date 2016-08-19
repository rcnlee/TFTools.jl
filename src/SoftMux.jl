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
    expand_dims, tile, l2_normalize, cond, truncated_normal, zeros_like

"""
Soft multiplexer (selector) component that can be learned using gradient
descent
"""
type SoftMux <: AbstractMux
    n_muxinput::Int64
    hidden_units::Vector{Int64}
    muxin::Tensor
    muxselect::Tensor
    softness::Tensor
    nn::ReluStack
    weight::Variable
    nnout::Tensor
    softout::Tensor
    harden::Tensor
    hardselect::Tensor
    hardout::Tensor
    override::Int64
end

using Debug
#"""
#harden::Bool that switches mux output to hard
#"""
@debug function SoftMux(n_muxinput::Int64, 
    hidden_units::Vector{Int64}, 
    muxin::Tensor, 
    muxselect::Tensor,
    harden::Tensor,
    softness::Tensor=constant(1.0); 
    override::Int64=0)

    relustack = ReluStack(muxselect, hidden_units)
    reluout = out(relustack)
    
    # last layer is softmax
    # softmax select over inputs
    n0 = get_shape(reluout)[end]
    n1 = n_muxinput 
    weight = Variable(truncated_normal(constant([n0, n1]), constant(0.0), constant(5e-2)))
    nnout = softmax(mul(softness, reluout * l2_normalize(weight, constant(1))))
    
    # mux output is the soft selected input
    if override > 0
        hardselect = tile(constant([override-1]), Tensor([size(nnout)[1]])) 
    else
        hardselect = arg_max(nnout, constant(1)) #hardened dense selected channel, 0-indexed
    end
    hardselect_1h = one_hot(hardselect, constant(n_muxinput))
    muxin_rank = ndims(muxin)
    if muxin_rank == 1 || muxin_rank == 2 #TODO: avoid switching if possible
        softout = reduce_sum(mul(muxin, nnout), constant(1))
        hardout = reduce_sum(mul(muxin, hardselect_1h), constant(1))
    elseif muxin_rank == 3
        softout = reduce_sum(mul3(muxin, nnout), constant(2))
        hardout = reduce_sum(mul3(muxin, hardselect_1h), constant(2))
    else
        error("Not supported! (rank=$(muxin_rank))")
    end

    SoftMux(n_muxinput, hidden_units, muxin, muxselect, softness, relustack, weight, 
        nnout, softout, harden, hardselect, hardout, override)
end

function mul3(X::Tensor, y::Tensor)
    n_time = get_shape(X)[2]
    tmp = expand_dims(y, constant(1))
    Y = tile(tmp, constant(Int32[1, n_time, 1]))
    out = mul(X, Y) #element-wise mul
    out
end

function out(mux::SoftMux) 
    cond_(mux.harden[1], ()->hardout(mux), ()->softout(mux))
end

function softselect(mux::SoftMux)
    mux.nnout
end

function softout(mux::SoftMux)
    mux.softout
end

function hardselect(mux::SoftMux)
    mux.hardselect
end

function hardout(mux::SoftMux)
    mux.hardout
end

function get_variables(mux::SoftMux)
    vars = vcat(mux.weight, get_variables(mux.nn))
    vars
end
