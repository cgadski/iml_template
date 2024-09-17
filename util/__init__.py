from matplotlib import pyplot as plt
import numpy as np

def shift(arr, shift, axis):
    result = np.roll(arr, shift, axis=axis)
    index = [slice(None),] * len(arr.shape)
    index[axis] = slice(None, shift)
    result[tuple(index)] = 0
    return result

def sierpinski_iterative(dim, iters=None):
    if iters is None:
        iters = dim * 2

    mat = np.zeros((dim, dim))
    mat[0, 0] = 1
    
    for i in range(iters):
        inputs = [shift(mat, 1, 1), shift(mat, 1, 0)]
        mat += np.logical_or(mat, np.logical_xor(*inputs))

    return mat

def show_matrix_big(mat):
    fig, ax = plt.subplots()
    ax.matshow(mat)
    ax.axis('off')
    fig.set_size_inches(8, 8)
    plt.show()

x = 100