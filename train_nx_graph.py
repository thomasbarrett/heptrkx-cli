#!/usr/bin/env python
import argparse
import os
import glob
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np

from nx_graph.utils_train import create_loss_ops
from nx_graph.utils_train import compute_matrics
from nx_graph.utils_train import load_config
from nx_graph.model import SegmentClassifier
from heptrkxcli.hitgraph import load_graph
from heptrkxcli.visualize import visualize_hitgraph
from heptrkxcli.triplet import node_parameters

def batch_iterator(base_dir, n_batch, test=False):
    '''
    Iterates through batches of size n_batch found at directory base_dir
    '''
    input_files = glob.glob(os.path.join(base_dir,'*'))
    input_files.sort()
    n_events = len(input_files) - 100
    batch_count = int(n_events/n_batch)    

    if test:
        input_graphs = []
        for file_index in range(0, 1):
            index = n_events + file_index
            (graph, truth, probability, triplets) = load_graph(input_files[index])
            input_graphs.append(graph)
            truth_values = truth
            triplet_list = triplets
            probability_list = probability
        yield 0, 0, input_graphs, truth_values, triplet_list, probability_list

    for batch_index in range(0, batch_count):
        input_graphs = []
        for file_index in range(0, 1):
            index = batch_index * n_batch + file_index
            (graph, truth, probability, triplets) = load_graph(input_files[index])
            input_graphs.append(graph)
            truth_values = truth
            triplet_list = triplets
            probability_list = probability
        yield batch_index, batch_count, input_graphs, truth_values, triplet_list, probability_list
    
    return

def main():
    
    # A bunch of configuration stuff to clean up...
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('name', nargs='?', default='unnamed')
    args = parser.parse_args()

    results_dir = 'results/{}'.format(args.name)
    os.makedirs(results_dir, exist_ok=True)
    config = load_config('configs/nxgraph_default.yaml')
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
    _, _, input_graphs, truth_values, triplets, probability = batch_iterator(base_dir, batch_size).__next__()
    input_ph = utils_tf.placeholders_from_data_dicts(input_graphs[0:1], force_dynamic_num_graphs=True)
    truth_ph = tf.placeholder(tf.float64, shape=[None])
    triplets_ph = tf.placeholder(tf.float64, shape=[4, None])

    # Here, we define our computational graphs.
    # - First, we compute the model output using the graph_nets library.
    # - Then, we compute our loss function only on edge features, where we utilize a log_loss
    #   function between the truth values and the model output. There is also some factor
    #   'num_processing_steps_tr' that describes the level of message passing that somehow
    #   plays into this. I need to figure out the details.
    # -  Finally, we will minimize training loss using the Adam Optimizer algorithm. 
    model_outputs = SegmentClassifier()(input_ph, num_processing_steps_tr)
    edge_losses = tf.losses.log_loss(tf.square(truth_ph), tf.transpose(model_outputs[-1].edges)[0])
    node_params = node_parameters(input_ph, model_outputs[-1].edges, triplets_ph)
    node_params_truth = node_parameters(input_ph, truth_ph, triplets_ph)
    node_params_error = tf.losses.mean_squared_error(node_params_truth, node_params)
    training_loss = edge_losses
    training_optimizer = tf.train.AdamOptimizer(0.001).minimize(training_loss)

    # Allows a graph containing `None` fields to be run in a Tensorflow
    # session. This is currently not needed since we have data for all
    # elements in the graph, including useless data for the global variable.
    input_ph = utils_tf.make_runnable_in_session(input_ph)

    # According to documentation, represent a connection between the client
    # program and a C++ runtime. See the following link for more information.
    # https://www.tensorflow.org/guide/graphs
    sess = tf.Session()
 
    # Create session saver
    saver = tf.train.Saver()
    saver.restore(sess, 'results/mrphi/models/model18.ckpt')

    # Our computation graph uses global variables, so we are required to
    # initialize them for the first pass. See the following link for more
    # information on Tensorflow variables
    # https://www.tensorflow.org/guide/variables
    # sess.run(tf.global_variables_initializer())

    output_index = 0 
    last_output = time.time()

    # We will iterate through our dataset many times to train.
    for iteration in range(0, num_training_iterations):
        
        # Iterate through all of the batches and retrieve batch data accordingly.
        for batch_index, batch_count, input_batch, truth_batch, triplets, probability in batch_iterator(base_dir, batch_size):
            
            # Turn our data dictionary into a proper graphs.GraphsTuple
            # object for use with graph_nets library. 
            input_graphs = utils_np.data_dicts_to_graphs_tuple(input_batch)

            # The utility function make_runnable_in_session to fix problems resulting from
            # None fields in graph.
            input_graphs = utils_tf.make_runnable_in_session(input_graphs)

            # Create a feed dictionary that properly maps graph properties.
            # Documentation states that this is only necessary in the case of
            # missing properties, but we will do it anyway just to be safe.
            feed_dict = utils_tf.get_feed_dict(input_ph, input_graphs)

            # We must pass both the input and target graphs into our computation
            # graph, so we update our feed dictionary with new properties using
            # the same method described above.

            feed_dict.update({truth_ph: probability})
            feed_dict.update({triplets_ph: triplets})

            # Run our computation graph using the feed_dictionary created above.
            # Currently, we appear to be computing multiple values... I need
            # to figure out what each of them means.
            train_values = sess.run({
                "step": training_optimizer,
                "loss": training_loss,
                "outputs": model_outputs
            }, feed_dict=feed_dict)

            # Compute the time lapse from last save-evaluate-visualize action
            current_time = time.time()
            output_time_lapse = current_time - last_output

            if output_time_lapse > 30:
                last_output = current_time

                # Create a feed dict with 10 training events. These events have not been
                # used during testing, so 
                
                _, _, input_batch, truth_batch, triplets, probability = batch_iterator(base_dir, 1, test=True).__next__()

                input_graphs = utils_np.data_dicts_to_graphs_tuple(input_batch)
                input_graphs = utils_tf.make_runnable_in_session(input_graphs)
                feed_dict = utils_tf.get_feed_dict(input_ph, input_graphs)
                feed_dict.update({truth_ph: probability})
                feed_dict.update({triplets_ph: triplets})

                train_values = sess.run({
                    "loss": training_loss,
                    "error": node_params_error,
                    "target": truth_ph,
                    "outputs": model_outputs
                }, feed_dict=feed_dict)

                cutoff_list = []
                purity_list = []
                efficiency_list = []

                # Compute purity and efficiency for every cutoff from 0 to 1 in steps of 0.01
                for filter_cutoff in np.linspace(0,1,100):
                    result = np.transpose(np.where(train_values['outputs'][-1].edges > filter_cutoff, 1, 0))[0]
                    correct = np.sum(np.where(np.logical_and(result == truth_batch, result == np.ones(result.shape)), 1, 0))
                    purity = correct / np.sum(result) if np.sum(result) != 0 else 1.0
                    purity_list.append(purity)
                    efficiency = correct/ np.sum(truth_batch)
                    efficiency_list.append(efficiency)
                    cutoff_list.append(filter_cutoff)

                # Create purity-efficiency plot and save to folder
                plt.figure()
                plt.plot(purity_list, efficiency_list)
                plt.axis([0, 1, 0, 1])
                plt.xlabel('Purity')
                plt.ylabel('Efficiency')
                os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
                plt.savefig(os.path.join(results_dir, 'figures/purity_vs_efficiency{:02d}.png'.format(output_index)))
                plt.close()


                # Write the purity-efficiency
                csv_path = os.path.join(results_dir, 'figures/purity_vs_efficiency{:02d}.csv'.format(output_index))
                with open(csv_path, 'w') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['cutoff','purity', 'efficiency'])
                    for (cutoff, purity, efficiency) in zip(cutoff_list, purity_list, efficiency_list):
                        csv_writer.writerow([cutoff, purity, efficiency])

                os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
                saver.save(sess, os.path.join(results_dir, 'models/model{}.ckpt'.format(output_index)))
                
                visualize_hitgraph(os.path.join(results_dir, 'images'), output_index, {
                    'nodes': input_batch[0]['nodes'],
                    'edges': truth_batch,
                    'senders': input_batch[0]['senders'],
                    'receivers': input_batch[0]['receivers']
                })

                visualize_hitgraph(os.path.join(results_dir, 'images'), output_index, {
                    'nodes': input_batch[0]['nodes'],
                    'edges': truth_batch,
                    'senders': input_batch[0]['senders'],
                    'receivers': input_batch[0]['receivers']
                })


                print('\repoch: {} progress: {:.4f} loss: {:.4f} error: {:.4f}'.format(iteration, batch_index/batch_count, train_values['loss'], train_values['error']))  

                output_index += 1


    sess.close()

if __name__ == "__main__":
    main()