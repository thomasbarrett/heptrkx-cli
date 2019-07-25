# Triplet Computation
Define a *sparse n-tensor* as two-tuple (indices, values) of parallel arrays where *indices* is an array of n-tuples specifying the location of an element and where *values* is an array of element values.

Define a *graph* as a four-tuple (nodes, edges, senders, receivers) where *nodes* is an array of size N containing node features, and *edges*, *senders*, *receivers* are parallel arrays of size E containing edge features, sending nodes indices, and recieving node indices respectively.

## Truth Coefficient Matrix
The truth coefficient matrix TCM is an N x N sparse 2-tensor where each element a_(i,j) in TCM equals the truth value t for the edge connecting the ith and jth node (n_i and n_j). If an edge does not exist between n_i and nodes<sub>j</sub> then the element is zero. Note that each column of the TCM represents the outgoing edges for n_i and that each row of TCM represents the incoming edges for n_j. This *sparse 2-tensor* can be constructed directly from our dataset. We define our *indices* as the transpose of our E x 2 senders and recievers arrays. We define our *values* as the `truth` feature for each edge.

## Truth Coefficient Tensor
The truth coefficient tensor TCT is a sparse N x N x N 3-tensor where each element a_(i,j,k) in TCT contains the product of the truth value from the ith incoming edge and the kth outgoing edge of node j. This tensor can be constructed by repeating our TCM N-times, transposing it to align the nth column of the TCM with the nth row, and performing elementwise multiplication by the transposed tensor.

## Triplet Parameter Tensor
The Triplet Parameter Tensor TPT is a N x N x N *sparse 3-tensor* where each element a_(i,j,k) in the TPT contains the helix parameter for the triple (n_i, n_j, n_k). This *sparse 3-tensor* can be constructed directly from our triplet parameter dataset since each element in the dataset contain a triplet of node indices and an associated radius.

## Node Parameter Regression
We can compute our node parameters by calculating the non-zero mean of each matrix in the TPT. This operation will output a size E vector. We can the train our edge classifier by regression using the node parameters computed from truth values output from the edge classifier. 


