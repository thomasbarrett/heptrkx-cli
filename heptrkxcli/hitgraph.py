import numpy as np

def save_graph(filename, graph):
    np.savez(filename,
        nodes=graph['nodes'],
        edges=graph['edges'],
        senders=graph['senders'],
        receivers=graph['receivers'])

def load_graph(filename):
    with np.load(filename, allow_pickle=True) as file:
        return {
            'nodes': file['nodes'],
            'edges': file['edges'],
            'senders': file['senders'],
            'receivers': file['receivers'],
            'globals': np.array([0.])
        }
