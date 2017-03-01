
# Tensor Ranks and Tensor Shapes:

TensorFlow programs use a tensor data structure to represent all data. 
You can think of a TensorFlow tensor as an n-dimensional array or list. A tensor has a static type and dynamic dimensions.


### Rank
In the TensorFlow system, tensors are described by a unit of dimensionality known as Rank. 
Tensor rank is the number of dimensions of the tensor. For example, consider the following table:

 ---------------------------------------------------------------------------------------------------------
| Rank  | Math Entity                      | Python Example                                               |
| ----- | -------------------------------- | ------------------------------------------------------------ |
| 0     | Scalar (magnitude only)          | s = 483                                                      |
| 1     | Vector (magnitude and direction) | v = [1.1, 2.2, 3.3]                                          |
| 2     | Matrix (table of numbers)        | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]                        |
| 3     | 3-Tensor (cube of numbers)       | t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] |
| n     | n-Tensor (you get the idea)      | ....                                                         |
 ----------------------------------------------------------------------------------------------------------


### Shape
The TensorFlow documentation uses three notational conventions to describe tensor dimensionality: rank, shape, and dimension number.
The following table shows how these relate to one another:

 -------------------------------------------------------------------------------------------- 
| Rank  | Shape              | Dimension Number   | Example                                 |
| ----- | ------------------ | ------------------ | --------------------------------------- |
| 0     | [ ]                | 0-D                | A 0-D tensor. A scalar.                 |
| 1     | [D0]               | 1-D                | A 1-D tensor with shape [5]             |
| 2     | [D1, D2]           | 2-D                | A 2-D tensor with shape [3, 4]          |
| 3     | [D1, D2, D3]       | 3-D                | A 3-D tensor with shape [1, 2, 3]       |
| n     | [D0, D1, ... Dn-1] | n-D                | A tensor with shape [D0, D1, ... Dn-1]  |
 --------------------------------------------------------------------------------------------
