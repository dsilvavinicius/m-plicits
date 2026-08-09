#from NeuralImplicit import NeuralImplicit
import torch
from torch import nn
import numpy as np
import struct
import argparse
import os

def format_matrix(matrix, transpose, reshape=True):
    
    matrix = matrix.cpu().numpy().transpose() if transpose else matrix.cpu().numpy()
    
    if reshape:
        matrix = np.squeeze(matrix.reshape(-1, 1))
    
    return matrix

# Returns biases (tranpose=True to transpose the resulting matrix').
def read_network(state_dict, transpose=False, decomposed=False):

    weights = np.empty((0,))
    biases = np.empty((0,))
    
    matrices = list(state_dict.values())

    # Input layer
    tmp_mat = format_matrix(matrices[0], transpose)
    weights = np.concatenate((weights, tmp_mat))
    tmp_mat = format_matrix(matrices[1], transpose, False)
    biases = np.concatenate((biases, tmp_mat))

    # Hidden layers
    if decomposed:
        for i in range(2, len(matrices) - 2, 3):
            U = format_matrix(matrices[i], transpose)
            V = format_matrix(matrices[i + 1], transpose)
            bias = format_matrix(matrices[i + 2], transpose, False)

            weights = np.concatenate((weights, U))
            weights = np.concatenate((weights, V))
            biases = np.concatenate((biases, bias))
    else:
        for i in range(2, len(matrices) - 2, 2):
            U = format_matrix(matrices[i], transpose)
            bias = format_matrix(matrices[i + 1], transpose, False)

            weights = np.concatenate((weights, U))
            biases = np.concatenate((biases, bias))

    # Output layer
    tmp_mat = format_matrix(matrices[-2], transpose)
    weights = np.concatenate((weights, tmp_mat))
    tmp_mat = format_matrix(matrices[-1], transpose, False)
    biases = np.concatenate((biases, tmp_mat))

    return {'weights': weights, 'biases': biases}

def write_file(floats, filename):
    print('Writting binary file ' + filename + ' (' + str(len(floats)) + ' elements).')
    s = struct.pack('f'*len(floats), *floats)
    f = open(filename,'wb')
    f.write(s)
    f.close()

parser = argparse.ArgumentParser(epilog='Creates a linearized binary matrix file from a .pth network. The result is always row-major.')
parser.add_argument('-f', '--file', help="Path to the .pth file.", required=True)
parser.add_argument('-t', '--tranpose', help="Indicates to transpose the output matrix.", action="store_true")
parser.add_argument('-d', '--decomposed', help="Indicates that the network is decomposed (i.e. hidden layers are matrix pairs, with the first one having no bias).",
action="store_true")
args = parser.parse_args()

filename = args.file
transpose = args.tranpose
decomposed = args.decomposed

print(os.getcwd())

state_dict = torch.load(filename)

print('State Dict keys and shapes:')
for key in state_dict.keys():
    print(f"{key}: {state_dict[key].size()}")

print('State Dict values: ' + str(state_dict))

weights_and_biases = read_network(state_dict, transpose, decomposed)

weights = weights_and_biases['weights']
print('Weights:\n' + str(weights))

biases = weights_and_biases['biases']
print('Biases:\n' + str(biases))

filename = os.path.splitext(filename)[0]
write_file(weights, filename + '_weights.bin')
write_file(biases, filename + '_biases.bin')