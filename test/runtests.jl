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

using TFTools
using Datasets
using Base.Test
using TensorFlow
import TensorFlow: DT_FLOAT32
import TensorFlow.API: l2_loss, AdamOptimizer, cast, round_

# Parameters
const LEARNING_RATE = 0.001
const TRAINING_EPOCHS = 50
const BATCH_SIZE = 100
const DISPLAY_STEP = 1

# Network parameters
const HIDDEN_UNITS = [100, 100] 

function test_softmux(datname::AbstractString="bin_small",
    featfile::AbstractString="feats", labelfile::AbstractString="labels_mux";
    labelfield::AbstractString="x1")

    Dfeats = dataset(datname, featfile)
    Dlabels = dataset(datname, labelfile)
    n_feats = ncol(Dfeats)
    @assert nrow(Dfeats) == nrow(Dlabels)

    data_set = TFDataset(Dfeats, Dlabels[symbol(labelfield)])

    # Construct model
    muxselect = Placeholder(DT_FLOAT32, [-1, n_feats])
    muxin = Placeholder(DT_FLOAT32, [-1, n_feats])
    mux = Softmux(n_feats, n_feats, HIDDEN_UNITS, Tensor(muxin), Tensor(muxselect))
    pred = out(mux) 
    labels = Placeholder(DT_FLOAT32, [-1]) #or should this be [-1, 1]?
    
    # Define loss and optimizer
    cost = l2_loss(pred - labels) # Squared loss
    optimizer = minimize(AdamOptimizer(LEARNING_RATE), cost) # Adam Optimizer
    
    # Initializing the variables
    init = initialize_all_variables()
    
    sess = Session()
    try
        run(sess, init)
        
        # Training cycle
        for epoch in 1:TRAINING_EPOCHS
            avg_cost = 0.0
            total_batch = div(num_examples(data_set), BATCH_SIZE)
        
            # Loop over all batches
            for i in 1:total_batch
                batch_xs, batch_ys = next_batch(data_set, BATCH_SIZE)
                # Fit training using batch data
                run(sess, optimizer, FeedDict(muxin => batch_xs, muxselect => batch_xs, labels => batch_ys))
                # Compute average loss
                batch_average_cost = run(sess, cost, FeedDict(muxin => batch_xs, muxselect => batch_xs, labels => batch_ys))
                avg_cost += batch_average_cost / (total_batch * BATCH_SIZE)
            end
        
            # Display logs per epoch step
            if epoch % DISPLAY_STEP == 0
                println("Epoch $(epoch)  cost=$(avg_cost)")
            end
        end
        println("Optimization Finished")
        
        # Test model
        correct_prediction = (round_(pred) == labels)
        # Calculate accuracy
        accuracy = mean(cast(correct_prediction, DT_FLOAT32))
        acc = run(sess, accuracy, FeedDict(muxin => data_set.X, muxselect => data_set.X, labels => data_set.Y))
        println("Accuracy:", acc)
    finally
        close(sess)
    end
    
end #module
