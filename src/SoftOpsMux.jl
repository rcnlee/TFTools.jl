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

"""
Creates a combination component of OpsBlock followed by Softmux
"""
type SoftOpsMux <: AbstractMux
    opsblock::OpsBlock
    softmux::SoftMux
end

function SoftOpsMux(op_inputs::Tuple{Vararg{Tensor}}, ops_list::Vector{Function},
    hidden_units::Vector{Int64}, muxselect::Tensor, softness::Tensor=constant(1.0))

    #ops
    opsblock = OpsBlock(op_inputs, ops_list)

    #mux
    n_muxinput = length(ops_list)
    softmux = SoftMux(n_muxinput, hidden_units, out(opsblock), muxselect, softness)

    SoftOpsMux(opsblock, softmux)
end

#forward these to opsblock
num_ops(opsmux::SoftOpsMux) = num_ops(opsmux.opsblock)

#AbstractMux interface --- forward these to softmux
out(opsmux::SoftOpsMux) = out(opsmux.softmux)
softout(opsmux::SoftOpsMux) = softout(opsmux.softmux)
softselect(opsmux::SoftOpsMux) = softselect(opsmux.softmux)
hardout(opsmux::SoftOpsMux) = hardout(opsmux.softmux)
hardselect(opsmux::SoftOpsMux) = hardselect(opsmux.softmux)

