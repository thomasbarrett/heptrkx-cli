import numpy as np
import os.path as path
import tensorflow as tf

def save_graph(filename, graph):
    np.savez(filename,
        nodes=graph['nodes'],
        edges=graph['edges'],
        senders=graph['senders'],
        receivers=graph['receivers'])

def load_graph_old(filename):
    with np.load(filename, allow_pickle=True) as file:
        return {
            'nodes': file['nodes'].astype(np.float32),
            'edges': file['edges'].astype(np.float32),
            'senders': file['senders'].astype(np.float32),
            'receivers': file['receivers'].astype(np.float32),
            'globals': np.array([0.]).astype(np.float32)
        }

def load_graph(filename):
	# subdirectories for edge, path, and triplet data
	node_path = path.join(filename, 'nodes')
	edge_path = path.join(filename, 'edges')
	triplet_path = path.join(filename, 'triplets')

	# load node data for input graph
	nodes = np.transpose(np.vstack((
		np.load(path.join(node_path, 'r.npy')),
		np.load(path.join(node_path, 'phi.npy')),
		np.load(path.join(node_path, 'z.npy'))
	)))

	# load edge data for input graph
	edges = np.transpose(np.vstack((
		np.load(path.join(edge_path, 'dr.npy')),
		np.load(path.join(edge_path, 'dphi.npy')),
		np.load(path.join(edge_path, 'dz.npy')),
		np.load(path.join(edge_path, 'mrphi.npy')),
	)))

	# load sender data for input graph
	senders = np.load(path.join(edge_path, 'senders.npy'))

	# load receiver data for input graph
	receivers = np.load(path.join(edge_path, 'receivers.npy'))

	# load triplet data for loss function
	
	triplets = np.vstack((
		np.load(path.join(triplet_path, 'incoming_edge_index.npy')),
		np.load(path.join(triplet_path, 'node_index.npy')),
		np.load(path.join(triplet_path, 'outgoing_edge_index.npy')),
		np.load(path.join(triplet_path, 'radius.npy'))
	))

	# load truth value for loss function
	truth = np.load(path.join(edge_path, 'truth.npy'))
	probability = np.load(path.join(edge_path, 'probability.npy'))

	graph = {
		'nodes': nodes,
		'edges': edges,
		'senders': senders,
		'globals': np.array([0.]),
		'receivers': receivers
	}

	return (graph, truth, probability, triplets)
