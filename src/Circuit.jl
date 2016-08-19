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
using TensorFlow.API: gradients

type Circuit
    blks::Vector{AbstractMux}
    opnames::Vector{Vector{ASCIIString}}
end

function hardselect_tensor(ckt::Circuit)
    hardselects = [hardselect(blk) for blk in ckt.blks] 
    hardtensor = transpose_(pack(Tensor(hardselects)))
    hardtensor
end

function softselect_tensor(ckt::Circuit)
    softsels = [softselect(blk) for blk in ckt.blks]
    softtensor = Tensor(softsels)
    softtensor
end

#"""
#Output a simple string concatenated by delimiter
#"""
using Debug
@debug function simplestring(sess::Session, ckt::Circuit, fd::FeedDict=FeedDict(); 
    order::Vector{Int64}=Int64[], delim::ASCIIString="_")
    
    hs = run(sess, hardselect_tensor(ckt), fd)
    n_examples, n_selectors = size(hs)

    #@bp 

    if isempty(order) 
        order = collect(1:n_selectors)
    end

    stringout = ASCIIString[]
    for i = 1:n_examples
        A = [ckt.opnames[j][hs[i,j]+1] for j in order] #x+1 for python encoding
        s = join(A, delim)
        push!(stringout, s)
    end
    stringout
end

function topstrings{S<:AbstractString}(stringout::Vector{S}, n_top::Int64)
    cmap = countmap(stringout)
    cmap1 = collect(cmap)
    tmp = sort(collect(cmap1), by=kv->kv[2], rev=true)
    n_top = min(n_top, length(tmp)) #don't exceed length
    topentries = tmp[1:n_top] #take top n
    topentries = map(x->(x...), topentries)
    topentries
end

"""
Executes each blk individually in a loop.  Good for pin pointing exceptions when debugging.
"""
function exec_blks(sess::Session, ckt::Circuit, fd::FeedDict=FeedDict();
    verbose::Bool=true)
    results = Any[]
    for i = 1:endof(ckt.blks)
        verbose && println("blk=$(ckt.blks[i])")
        result = run(sess, out(ckt.blks[i]), fd) 
        push!(results, result)
    end
    results
end

function gradient_tensor(target::Tensor, vars::Variable...)
    Tensor(gradients(target, Tensor([vars...]))[1])
end

function softselect_by_example(sess::Session, ckt::Circuit, fd::FeedDict=FeedDict())
    softsel = run(sess, softselect_tensor(ckt), fd)
    byexample = Any[]
    n_examples = size(softsel[1], 1) #rows of the first blk
    for i = 1:n_examples
        A = [softsel[j][i,:] for j=1:length(ckt.blks)]
        push!(byexample, A)
    end
    byexample
end

