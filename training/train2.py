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

from heptrkxcli.model import SegmentClassifier
from heptrkxcli.hitgraph import load_graph
from heptrkxcli.visualize import visualize_hitgraph
from heptrkxcli.triplet import node_parameters

def batch_iterator(base_dir, n_batch, test=False):
    '''
    Iterates through batches of size n_batch found at directory base_dir
    '''
    input_files = glob.glob(os.path.join(base_dir,'*'))
    input_files.sort()
    n_events = len(input_files)
    n_train = len(input_files) - 100
    n_test = n_events - n_train
    batch_count = int(n_events/n_batch)    

    if test:
        for batch_index in range(0, n_test):
            index = n_train + batch_index
            (graph, truth, triplets) = load_graph(input_files[index])
            input_graphs = [graph]
            truth_values = truth
            triplet_list = triplets
            yield batch_index, n_test, input_graphs, truth_values, triplet_list
    else:
        for batch_index in range(0, batch_count):
            index = batch_index
            (graph, truth, triplets) = load_graph(input_files[index])
            input_graphs = [graph]
            truth_values = truth
            triplet_list = triplets
            yield batch_index, batch_count, input_graphs, truth_values, triplet_list
    return

def main():
    
    # A bunch of configuration stuff to clean up...
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg('name', nargs='?', default='unnamed')
    args = parser.parse_args()
    output_dir = 'results/{}'.format(args.name)
    os.makedirs(output_dir, exist_ok=True)

    # hardcode options for now...
    base_dir                = '../events'
    log_every_seconds       = 60
    batch_size              = 1
    num_training_iterations = 5
    num_processing_steps    = 5
    learning_rate           = 0.001
    restore_path            = None
    restore_index           = 0

    # Start to build tensorflow sessions
    tf.reset_default_graph()

    # Creates a placeholder for training examples. The placeholders define a
    # slot for training examples given in feed dict to be assigned. We create
    # graphs.GraphsTuple placeholders using the graph_nets utility functions.
    # They are automatically generated from the first graph in the first batch.
    # By assigning force_dynamic_num_graphs=True, we ensure the the placeholders
    # accepts graphs of any size.
    _, _, input_graphs, truth_values, triplets = batch_iterator(base_dir, batch_size).__next__()
    input_ph = utils_tf.placeholders_from_data_dicts(input_graphs[0:1], force_dynamic_num_graphs=True)
    target_ph = tf.placeholder(tf.float64, shape=[None])
    triplets_ph = tf.placeholder(tf.float64, shape=[4, None])

    # Define output and loss functions for testing.
    # Here, we are only interested in the final outout of the model. Thus, we do not
    # consider any of the intermediate 'core' outputs in our loss function. 
    output_ops = SegmentClassifier()(input_ph, num_processing_steps)
    output_op = output_ops[-1]
    node_params_truth = 10000 * node_parameters(input_ph, target_ph, triplets_ph)
    node_params = 10000 * node_parameters(input_ph, output_op.edges, triplets_ph)
    node_param_loss_op = tf.losses.mean_squared_error(node_params_truth, node_params)
    binary_classifier_loss_op = tf.losses.log_loss(target_ph, tf.squeeze(output_op.edges))
    loss_op = node_param_loss_op + binary_classifier_loss_op
    
    # The training optimizer
    training_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    # According to documentation, represent a connection between the client
    # program and a C++ runtime. See the following link for more information.
    # https://www.tensorflow.org/guide/graphs
    sess = tf.Session()
    saver = tf.compat.v1.train.Saver()
    # normalize gradient 
    # - gradients of both loss
    # - final gradient is ecpsilon times sum or nornalized gradients
    # - 

    # If a restore_path is give, restore our session from the tensorflow checkpoint
    # at the path. Otherwise, initialize global variables
    if restore_path == None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, restore_path)

    # Our computation graph uses global variables, so we are required to
    # initialize them for the first pass. See the following link for more
    # information on Tensorflow variables
    # https://www.tensorflow.org/guide/variables

    output_index = restore_index
    last_output = time.time()

    test_loss = []
    test_classifier_loss = []
    test_parameter_loss = []
    test_epoch = []

    train_loss = []
    train_classifier_loss = []
    train_parameter_loss = []
    train_epoch = []

    # We will iterate through our dataset many times to train.
    for iteration in range(0, num_training_iterations):
        
        # Iterate through all of the batches and retrieve batch data accordingly.
        for batch_index, batch_count, input_batch, labels, triplets in batch_iterator(base_dir, batch_size):
            
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

            feed_dict.update({target_ph: labels})
            feed_dict.update({triplets_ph: triplets})

            # Run our computation graph using the feed_dictionary created above.
            # Currently, we appear to be computing multiple values... I need
            # to figure out what each of them means.
            train_values = sess.run({
                "step": training_optimizer,
                "classifier_loss": binary_classifier_loss_op,
                "node_param_loss": node_param_loss_op,
                "loss": loss_op,
                "outputs": output_op
            }, feed_dict=feed_dict)

            # Compute the time lapse from last save-evaluate-visualize action
            current_time = time.time()
            output_time_lapse = current_time - last_output

            if output_time_lapse > log_every_seconds:
                last_output = current_time

                batch_labels = np.array([])
                batch_predictions = np.array([])
                classifier_loss = []
                node_param_loss = []
                loss = []

                # Run edge classifier on all test samples and 
                for _, _, input_batch, labels, triplets in batch_iterator(base_dir, 1, test=True):
                    input_graphs = utils_np.data_dicts_to_graphs_tuple(input_batch)
                    input_graphs = utils_tf.make_runnable_in_session(input_graphs)

                    feed_dict = utils_tf.get_feed_dict(input_ph, input_graphs)
                    feed_dict.update({target_ph: labels})
                    feed_dict.update({triplets_ph: triplets})

                    train_values = sess.run({
                        "classifier_loss": binary_classifier_loss_op,
                        "node_param_loss": node_param_loss_op,
                        "loss": loss_op,
                        "target": target_ph,
                        "predictions": output_op.edges,
                    }, feed_dict=feed_dict)

                    batch_labels = np.concatenate((batch_labels, labels), axis=0)
                    batch_predictions = np.concatenate((batch_predictions, np.squeeze(train_values['predictions'])), axis=0)
                    classifier_loss.append(train_values['classifier_loss'])
                    node_param_loss.append(train_values['node_param_loss'])
                    loss.append(train_values['loss'])

                # The threshold values to distinguish between true and false
                # edges. We want to generate thresholds from 0 to 1.
                thresholds = np.linspace(0,1,101)
                
                # Create tensorflow operation to calculate precision and recall
                precision_op = tf.compat.v1.metrics.precision_at_thresholds(batch_labels, batch_predictions,thresholds.tolist())    
                recall_op = tf.compat.v1.metrics.recall_at_thresholds(batch_labels, batch_predictions, thresholds.tolist())

                # Precision and Recall calculations create local variables
                # that must be initialized before calculation.
                sess.run(tf.local_variables_initializer())
                precision = sess.run(precision_op)[0]
                recall = sess.run(recall_op)[0]

                # Plot purity and efficiency graphs
                plot_precision_and_recall(
                    output_dir,
                    output_index,
                    thresholds,
                    precision,
                    recall
                )

                # Save the trained model
                # os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
                # saver.save(sess, os.path.join(output_dir, 'models/model{}.ckpt'.format(output_index)))
                    
                # We choose our cutoff based on the following:
                #   First - we only consider cutoffs that result in at least 95% recall
                #   Second - we maximize distance from origin, weighting precision and recall equally
                filtered_recall = recall[recall > 0.95]
                filtered_precision = precision[recall > 0.95]
                filtered_thresholds = thresholds[recall > 0.95]
                index = np.argmax(filtered_precision ** 2 + filtered_recall ** 2)

                threshold = filtered_thresholds[index]

                # Create an output graph out of the resulting graph
                output_graph = {
                    'nodes': input_batch[0]['nodes'],
                    'edges': np.squeeze(np.take(input_batch[0]['edges'], batch_predictions > threshold, axis=0)),
                    'senders': np.squeeze(np.take(input_batch[0]['senders'], batch_predictions > threshold, axis=0)),
                    'receivers': np.squeeze(np.take(input_batch[0]['receivers'], batch_predictions > threshold, axis=0))
                }

                classifier_loss = np.mean(np.array(classifier_loss))
                node_param_loss = np.mean(np.array(node_param_loss))
                loss = np.mean(np.array(loss))

                test_loss.append(loss)
                test_classifier_loss.append(classifier_loss)
                test_parameter_loss.append(node_param_loss)
                test_epoch.append(iteration + batch_index/batch_count)

                train_loss.append(train_values['loss'])
                train_classifier_loss.append(train_values['classifier_loss'])
                train_parameter_loss.append(train_values['node_param_loss'])
                train_epoch.append(iteration + batch_index/batch_count)

                plt.figure()
                plt.plot(test_epoch, test_loss, label="Testing Loss")
                plt.plot(train_epoch, train_loss, label="Training Loss")
                plt.plot(test_epoch, test_classifier_loss, label="Testing Classification Loss")
                plt.plot(train_epoch, train_classifier_loss, label="Training Classification Loss")
                plt.plot(test_epoch, test_parameter_loss, label="Testing Parameter Loss")
                plt.plot(train_epoch, train_parameter_loss, label="Training Parameter Loss")
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss vs. Epoch')
                plt.axis([0, 5, 0, 0.6])
                plt.legend()
                os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'figures/loss.png'))
                plt.close()

                # Print training progress information
                print('\repoch: {} progress: {:.4f} classifier loss: {:.4f} node parameter loss: {:.4f} precision: {:.4f} recall: {:.4f} threshold: {:.4f}'.format(
                    iteration, batch_index/batch_count, classifier_loss, node_param_loss,
                    filtered_precision[index], filtered_recall[index], filtered_thresholds[index])
                )  

                output_index += 1

    sess.close()

def visualize_processed_hitgraphs(input_batch, labels, train_values, output_index):
    visualize_hitgraph(os.path.join(output_dir, 'images'), output_index, {
        'nodes': input_batch[0]['nodes'],
        'edges': labels,
        'senders': input_batch[0]['senders'],
        'receivers': input_batch[0]['receivers']
    })

    # Find and visualize all false positive edges in graph
    visualize_hitgraph(os.path.join(output_dir, 'output'), output_index, {
        'nodes': input_batch[0]['nodes'],
        'edges': train_values['outputs'].edges,
        'senders': input_batch[0]['senders'],
        'receivers': input_batch[0]['receivers']
    })

    # Find and visualize all false positive edges in graph
    visualize_hitgraph(os.path.join(output_dir, 'false_positive'), output_index, {
        'nodes': input_batch[0]['nodes'],
        'edges': train_values['false_positive'],
        'senders': input_batch[0]['senders'],
        'receivers': input_batch[0]['receivers']
    })

    # Find and visualize all false negative edges in graph
    visualize_hitgraph(os.path.join(output_dir, 'false_negative'), output_index, {
        'nodes': input_batch[0]['nodes'],
        'edges': train_values['false_negative'],
        'senders': input_batch[0]['senders'],
        'receivers': input_batch[0]['receivers']
    })

def plot_precision_and_recall(output_dir, output_index, cutoff_list, precision_list, recall_list):
    plt.figure()
    plt.plot(precision_list, recall_list)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision - Recall')
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'figures/precision_vs_recall{:02d}.png'.format(output_index)))
    plt.close()

    # Write the purity-efficiency
    csv_path = os.path.join(output_dir, 'figures/precision_vs_recall{:02d}.csv'.format(output_index))
    with open(csv_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['cutoff','precision', 'recall'])
        for (cutoff, precision, recall) in zip(cutoff_list, precision_list, recall_list):
            csv_writer.writerow([cutoff, precision, recall])

if __name__ == "__main__":
    main()