
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

#supplement to TensorFlow.jl -- remove after TensorFlow.jl update
using PyCall

@pyimport tensorflow as tf
using TensorFlow.CoreTypes

one_hot(indices::Union{AbstractTensor,Void}, depth::Union{AbstractTensor,Void}, on_value::Union{AbstractTensor,Void}=nothing, off_value::Union{AbstractTensor,Void}=nothing, axis::Union{AbstractTensor,Void}=nothing, dtype::Dtype=DT_FLOAT32, name::Union{AbstractString,Void}=nothing) = Tensor(tf.one_hot(;Dict(:indices=>indices, :depth=>depth, :on_value=>on_value, :off_value=>off_value, :axis=>axis, :dtype=>dtype, :name=>name)...))

export one_hot

function cond_(pred::Union{AbstractTensor,Void}, fn1::Function,
    fn2::Function, name::Union{AbstractString,Void}=nothing) 
    
    fun = pyeval("""lambda f: lambda: f()""")
    Tensor(tf.cond(;Dict(:pred=>pred, :fn1=>fun(fn1), :fn2=>fun(fn2), :name=>name)...))
end

export cond_
