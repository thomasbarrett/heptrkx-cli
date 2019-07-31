import tensorflow as tf
from heptrkxcli.hitgraph import load_graph
import numpy as np

def truth_coefficient_matrix(n_nodes, senders, receivers, truth):
    indices = tf.transpose(tf.stack([senders, receivers], axis=0))

    truth_matrix = tf.SparseTensor(
                        indices=tf.cast(indices, tf.int64),
                        values=tf.reshape(truth, [tf.shape(truth)[0]]),
                        dense_shape=[n_nodes] * 2)

    return tf.sparse.reorder(truth_matrix)

def triplet_parameter_tensor(n_nodes, edgei, node, edgeo, truth, parameters):
    # Using the edge index tensors, search for the truth values found at those indices.
    # Since our graphs are so large, we must using map_fn to search for our edge truths
    # in the (index, value) data structure maintained by SparseTensor. Once incoming
    # and outgoing truth values are located for each triplet, take the elementwise product
    # of incoming truth value, outgoing truth value, and helix parameter to get the
    # final value for the entry in our triplet tensor.
    triplet_edge1_truth = tf.gather(truth, edgei)
    triplet_edge2_truth = tf.gather(truth, edgeo)
    triplet_edges_truth = tf.multiply(triplet_edge1_truth, triplet_edge2_truth)
    triplet_values = tf.multiply(tf.squeeze(triplet_edges_truth), parameters)
    print(triplet_values.shape)
    # Construct sparse tensor with newly computed triplet values
    indices = tf.stack([edgei, node, edgeo], axis=1)
    triplet_tensor = tf.SparseTensor(indices=indices, values=triplet_values, dense_shape=[n_nodes] * 3)
    return tf.sparse.reorder(triplet_tensor)

def node_parameters(graph, truth, triplets):

    nodes = graph.nodes
    edges = graph.edges
    senders = graph.senders
    receivers = graph.receivers
    n_nodes = tf.shape(nodes)[0]

    edgei = tf.cast(triplets[0], tf.int64)
    node = tf.cast(triplets[1], tf.int64)
    edgeo = tf.cast(triplets[2], tf.int64)
    radius = tf.cast(triplets[3], tf.float64)
   
    triplet_tensor = triplet_parameter_tensor(n_nodes, edgei, node, edgeo, tf.cast(truth, tf.float64), radius)
    return triplet_tensor.values

    triplet_sum = tf.sparse_reduce_sum(triplet_tensor, axis=[0,2])

    triplet_count = tf.dtypes.cast(
                        tf.math.bincount(
                            tf.cast(node, tf.int32),
                        ),
                        tf.float64
                    )
                    
    triplet_count_padded = tf.pad(triplet_count, [[0, n_nodes - tf.shape(triplet_count)[0]]])
    triplet_count_padded_safe = tf.where_v2(
                                    tf.equal(triplet_count_padded, 0.0),
                                    tf.ones(tf.shape(triplet_count_padded), tf.float64),
                                    triplet_count_padded)

    triplet_average = tf.divide(triplet_sum, triplet_count_padded_safe)
    return triplet_average


def edge_parameters(graph, truth, triplets):
    nodes = graph.nodes
    edges = graph.edges
    senders = graph.senders
    receivers = graph.receivers
    n_nodes = tf.shape(nodes)[0]

    edgei = tf.cast(triplets[0], tf.int64)
    node = tf.cast(triplets[1], tf.int64)
    edgeo = tf.cast(triplets[2], tf.int64)
    radius = tf.cast(triplets[3], tf.float64)
   
    triplet_tensor = triplet_parameter_tensor(n_nodes, edgei, node, edgeo, tf.cast(truth, tf.float64), radius)
    return triplet_tensor.values

    triplet_sum = tf.sparse_reduce_sum(triplet_tensor, axis=[0,2])

    triplet_count = tf.dtypes.cast(
                        tf.math.bincount(
                            tf.cast(node, tf.int32),
                        ),
                        tf.float64
                    )
                    
    triplet_count_padded = tf.pad(triplet_count, [[0, n_nodes - tf.shape(triplet_count)[0]]])
    triplet_count_padded_safe = tf.where_v2(
                                    tf.equal(triplet_count_padded, 0.0),
                                    tf.ones(tf.shape(triplet_count_padded), tf.float64),
                                    triplet_count_padded)

    triplet_average = tf.divide(triplet_sum, triplet_count_padded_safe)
    return triplet_average
