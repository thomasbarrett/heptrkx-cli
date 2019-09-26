"""
This file contains code for iterating through tracks in a graph.

These functions should be applied to the output of the edge classifier
network in order to obtain the list of tracks corresponding to an
individual particle. 
"""
import numpy as np

def track_starts(graph):
    """
    Return a numpy array of all nodes with an outgoing edge but no incoming edges.
    These hits are the ones which start a track.
    """
    nodes = graph['nodes']
    senders = graph['senders']
    receivers = graph['receivers']

    node_indices = np.arange(nodes.shape[0])
    node_indices = node_indices[np.logical_not(np.isin(node_indices, receivers))]
    node_indices = node_indices[np.isin(node_indices, senders)]
    return node_indices.tolist()

def track_iterator_from_root(graph, root, stack=[]):
    """
    Iterate over all tracks in in the graph through a depth first traversal
    Note that this function assumes that there are no loops in the graph.
    """

    # Retrieve nodes, senders, and recievers from graph object. This function
    # currently only supports data_dict type graphs
    nodes = graph['nodes']
    senders = graph['senders']
    receivers = graph['receivers']
    outgoing_edges = np.argwhere(senders == root).tolist()
   
    if (len(outgoing_edges) != 0):
        for outgoing_edge in outgoing_edges:
            node = receivers[outgoing_edge] 
            stack.append(root)
            yield from track_iterator_from_root(graph, node, stack=stack)
            stack.pop()
    else:
        yield np.asarray(stack)

def track_iterator(graph):
    """
    This is the most convenient way to get contiguous tracksc in the graph.
    In order to get a list of tracks rather than iterate through all tracks,
    list comprehension can be used as follows.
    
    Example:
    tracks = [track for track in track_iterator]
    """
    for track_start in track_starts(graph):
        yield from track_iterator_from_root(graph, root=track_start)
