#!/usr/bin/env python
import argparse
import os
import glob
import numpy as np

import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np

from nx_graph.utils_train import create_loss_ops
from nx_graph.utils_train import compute_matrics
from nx_graph.utils_train import load_config
from nx_graph.model import SegmentClassifier
from heptrkxcli.hitgraph import load_graph

def batch_iterator(base_dir, n_batch):
    '''
    Iterates through batches of size n_batch found at directory base_dir
    '''
    input_files = glob.glob(os.path.join(os.path.join(base_dir,'input'),'*'))
    target_files = glob.glob(os.path.join(os.path.join(base_dir,'target'),'*'))
    n_events = len(target_files)
    batch_count = int(n_events/n_batch)

    for batch_index in range(0, batch_count):
        for file_index in range(0, n_batch):
            input_graphs = []
            target_graphs = []
            index = batch_index * n_batch + file_index
            input_graph = load_graph(input_files[index])
            target_graph = load_graph(target_files[index])
            input_graphs.append(input_graph)
            target_graphs.append(target_graph)
        yield input_graphs, target_graphs
    
    return

def main():
    # A bunch of configuration stuff to clean up...
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('config',  nargs='?', default='configs/nxgraph_default.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    base_dir = config['data']['input_dir']
    config_tr = config['train']
    log_every_seconds       = config_tr['time_lapse']
    batch_size              = config_tr['batch_size']   # need optimization
    num_training_iterations = config_tr['iterations']
    iter_per_job            = config_tr['iter_per_job']
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    learning_rate = config_tr['learning_rate']
    output_dir = os.path.join(config['output_dir'], prod_name)

    # Start to build tensorflow sessions
    tf.reset_default_graph()

    # Creates a placeholder for training examples. The placeholders define a
    # slot for training examples given in feed dict to be assigned. We create
    # graphs.GraphsTuple placeholders using the graph_nets utility functions.
    # They are automatically generated from the first graph in the first batch.
    # By assigning force_dynamic_num_graphs=True, we ensure the the placeholders
    # accepts graphs of any size.
    input_graphs, target_graphs = batch_iterator(base_dir, batch_size).__next__()
    input_ph = utils_tf.placeholders_from_data_dicts(input_graphs[0:1], force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs[0:1], force_dynamic_num_graphs=True)

    # Here, we define our computational graphs.
    # - First, we compute the model output using the graph_nets library.
    # - Then, we compute our loss function only on edge features, where we utilize a log_loss
    #   function between the truth values and the model output. There is also some factor
    #   'num_processing_steps_tr' that describes the level of message passing that somehow
    #   plays into this. I need to figure out the details.
    # -  Finally, we will minimize training loss using the Adam Optimizer algorithm. 
    model_outputs = SegmentClassifier()(input_ph, num_processing_steps_tr)
    edge_losses = [tf.losses.log_loss(target_ph.edges, output.edges) for output in model_outputs]
    training_loss = tf.divide(tf.reduce_sum(edge_losses), num_processing_steps_tr)
    training_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(training_loss)

    # Allows a graph containing `None` fields to be run in a Tensorflow
    # session. This is currently not needed since we have data for all
    # elements in the graph, including useless data for the global variable.
    input_ph = utils_tf.make_runnable_in_session(input_ph)
    target_ph = utils_tf.make_runnable_in_session(target_ph)

    # According to documentation, represent a connection between the client
    # program and a C++ runtime. See the following link for more information.
    # https://www.tensorflow.org/guide/graphs
    sess = tf.Session()
 
    # Our computation graph uses global variables, so we are required to
    # initialize them for the first pass. See the following link for more
    # information on Tensorflow variables
    # https://www.tensorflow.org/guide/variables
    sess.run(tf.global_variables_initializer())

    # We will iterate through our dataset many times to train.
    for iteration in range(0, num_training_iterations):
        
        # Iterate through all of the batches and retrieve batch data accordingly.
        for input_batch, target_batch in batch_iterator(base_dir, batch_size):
            
            # Turn our data dictionary into a proper graphs.GraphsTuple
            # object for use with graph_nets library. 
            input_graphs = utils_np.data_dicts_to_graphs_tuple(input_batch)
            target_graphs = utils_np.data_dicts_to_graphs_tuple(target_batch)

            # The utility function make_runnable_in_session to fix problems resulting from
            # None fields in graph.
            input_graphs = utils_tf.make_runnable_in_session(input_graphs)
            target_graphs = utils_tf.make_runnable_in_session(target_graphs)

            # Create a feed dictionary that properly maps graph properties.
            # Documentation states that this is only necessary in the case of
            # missing properties, but we will do it anyway just to be safe.
            feed_dict = utils_tf.get_feed_dict(input_ph, input_graphs)

            # We must pass both the input and target graphs into our computation
            # graph, so we update our feed dictionary with new properties using
            # the same method described above.
            feed_dict.update(utils_tf.get_feed_dict(target_ph, target_graphs))
        
            # Run our computation graph using the feed_dictionary created above.
            # Currently, we appear to be computing multiple values... I need
            # to figure out what each of them means.
            train_values = sess.run({
                "step": training_optimizer,
                "target": target_ph,
                "loss": training_loss,
                "outputs": model_outputs
            }, feed_dict=feed_dict)

    sess.close()

if __name__ == "__main__":
    main()