from heptrkxcli.triplet import node_parameters
import tensorflow as tf
import numpy as np
import unittest

class TestNodeParameters(unittest.TestCase):

    def test_one_true_triplet(self):
        '''
        Tests the node parameter computation for the following graph with
        the edge weights [1, 1].

        [0]--0--[1]--1--[2]

        Under these circumstances, there is only one triple ((0, 1, 1), r).
        Since the weights of both edges are 1, the node parameters should
        simply be the [0, r, 0].
        '''
        graph = {
            'nodes': [[0,0,0], [0,0,0], [0,0,0]],
            'edges': [[0,0,0,0],[0,0,0,0]],
            'senders': [0, 1],
            'receivers': [1, 2]
        }
        truth = [1, 1]
        triplets = {
            'edgei': [0],
            'node': [1],
            'edgeo': [1],
            'radius': [5] 
        }
        with tf.Session().as_default():
            params = node_parameters(graph, truth, triplets).eval()
            self.assertTrue(np.array_equal(params, [0, 5, 0]))

    def test_two_true_triplets(self):
        '''
        Tests the node parameter computation for the following graph with
        the edge weights [1, 1, 1].

        [0]--0--[1]--1--[2]--2--[3]

        Under these circumstances, there are two triples ((0, 1, 1), r1) and
        ((1, 2, 2), r2). Since the weights of all edges are 1, the node
        parameters should be [0, r1, r2, 0].
        '''
        graph = {
            'nodes': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]],
            'edges': [[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            'senders': [0, 1, 2],
            'receivers': [1, 2, 3]
        }
        truth = [1, 1, 1]
        triplets = {
            'edgei': [0, 1],
            'node': [1, 2],
            'edgeo': [1, 2],
            'radius': [5, 10] 
        }
        with tf.Session().as_default():
            params = node_parameters(graph, truth, triplets).eval()
            self.assertTrue(np.array_equal(params, [0, 5, 10, 0]))

    def test_two_mixed_triplets(self):
        '''
        Tests the node parameter computation for the following graph with
        the edge weights [1, 1, 0].

        [0]--0--[1]--1--[2]--2--[3]
        '''
        graph = {
            'nodes': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]],
            'edges': [[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            'senders': [0, 1, 2],
            'receivers': [1, 2, 3]
        }
        truth = [1, 1, 0]
        triplets = {
            'edgei': [0, 1],
            'node': [1, 2],
            'edgeo': [1, 2],
            'radius': [5, 10] 
        }
        with tf.Session().as_default():
            params = node_parameters(graph, truth, triplets).eval()
            self.assertTrue(np.array_equal(params, [0, 5, 0, 0]))

    def test_two_nonbinary_triplets(self):
        '''
        Tests the node parameter computation for the following graph with
        the edge weights [1, 0.5, 1].

        [0]--0--[1]--1--[2]--2--[3]
        '''
        graph = {
            'nodes': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]],
            'edges': [[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            'senders': [0, 1, 2],
            'receivers': [1, 2, 3]
        }
        truth = [1, 0.5, 1]
        triplets = {
            'edgei': [0, 1],
            'node': [1, 2],
            'edgeo': [1, 2],
            'radius': [5, 10] 
        }
        with tf.Session().as_default():
            params = node_parameters(graph, truth, triplets).eval()
            self.assertTrue(np.array_equal(params, [0, 2.5, 5, 0]))

    def test_merge_true_triplets(self):
        '''
        Tests the node parameter computation for the following graph with
        the edge weights [1, 1, 1].

        [0]--0--|
               [2]--2--[3]
        [1]--1--|
        '''
        graph = {
            'nodes': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]],
            'edges': [[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            'senders': [0, 1, 2],
            'receivers': [2, 2, 3]
        }
        truth = [1, 1, 1]
        triplets = {
            'edgei': [0, 1],
            'node': [2, 2],
            'edgeo': [2, 2],
            'radius': [5, 10] 
        }
        with tf.Session().as_default():
            params = node_parameters(graph, truth, triplets).eval()
            self.assertTrue(np.array_equal(params, [0, 0, 7.5, 0]))


if __name__ == '__main__':
    unittest.main()
