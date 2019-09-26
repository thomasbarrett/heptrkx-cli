# System Dependencies
import glob
import os

# External Dependencies
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np

# Local Dependencies
from heptrkxcli.hitgraph import load_graph
from nx_graph.model import SegmentClassifier

def main():
    # Create a list of all events
    base_dir = '../events'
    input_files = glob.glob(os.path.join(base_dir,'*'))
    input_files.sort()
    message_passing_level = 5
    cutoff = 0.04

    # Load the first graph to create placeholders
    (graph0, truth0, triplets0) = load_graph(input_files[0])
    input_ph = utils_tf.placeholders_from_data_dicts([graph0], force_dynamic_num_graphs=True)
    label_ph = tf.placeholder(tf.float32, shape=[None])
    model_output = SegmentClassifier()(input_ph, message_passing_level)
    edge_output = tf.squeeze(tf.where_v2(model_output[-1].edges > cutoff, 1.0, 0.0))
    edge_output_equals_label = tf.where_v2(tf.equal(edge_output, label_ph), 1.0, 0.0)

    true_positive = tf.logical_and(
                            tf.cast(edge_output, tf.bool),
                            tf.cast(edge_output_equals_label, tf.bool)
                         )

    true_positive_count = tf.reduce_sum(tf.cast(true_positive, tf.float32))
    purity = true_positive_count / tf.reduce_sum(edge_output)
    efficiency = true_positive_count / tf.reduce_sum(label_ph)

    edge_loss = tf.losses.log_loss(tf.square(label_ph), tf.transpose(model_output[-1].edges)[0])
    training_optimizer = tf.train.AdamOptimizer(0.001).minimize(edge_loss)

    # Create Tensorflow Session and restore from file
    sess1 = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess1, 'results/mrphi/models/model20.ckpt')
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Iterate through each graph to process
    for index in range(0, len(input_files)):
        (graph, truth, triplets) = load_graph(input_files[index])

        # Create feed_dict for event 
        input_graphs_tuple = utils_np.data_dicts_to_graphs_tuple([graph])
        feed_dict = utils_tf.get_feed_dict(input_ph, input_graphs_tuple)
        feed_dict.update({label_ph: truth})

        # Run model on inputs
        output = sess1.run({
            "edge_output": edge_output,
            "edge_label": label_ph,
            "purity": purity,
            "efficiency": efficiency
        }, feed_dict=feed_dict)

        # Remove Edges Culled by First Step
        graph['edges'] = tf.boolean_mask(graph['edges'], tf.cast(output['edge_output'], tf.bool)).eval(session=sess1)
        graph['senders'] = tf.boolean_mask(graph['senders'], tf.cast(output['edge_output'], tf.bool)).eval(session=sess1)
        graph['receivers'] = tf.boolean_mask(graph['receivers'], tf.cast(output['edge_output'], tf.bool)).eval(session=sess1)
        truth = tf.boolean_mask(truth, tf.cast(output['edge_output'], tf.bool)).eval(session=sess1)

        # Create feed_dict for event 
        input_graphs_tuple = utils_np.data_dicts_to_graphs_tuple([graph])
        feed_dict = utils_tf.get_feed_dict(input_ph, input_graphs_tuple)
        feed_dict.update({label_ph: truth})

        # Run model on inputs
        output = sess.run({
            "step": training_optimizer,
            "loss": edge_loss,
            "outputs": model_output
        }, feed_dict=feed_dict)

        print('loss: {}'.format(output['loss']))

if __name__ == "__main__":
    main()