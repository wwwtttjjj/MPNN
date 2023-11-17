import torch

def bin3(targets):
    binary_map = torch.logical_and(targets[3] == targets[1],targets[1] == targets[2])
    return binary_map

def bin4(targets):
    binary_map = torch.logical_and(torch.logical_and(targets[4] == targets[1],targets[1] == targets[2]),targets[2] == targets[3])
    return binary_map


def bin5(targets):
    binary_map = torch.logical_and(torch.logical_and(torch.logical_and(targets[5] == targets[1],targets[1] == targets[2]),targets[2] == targets[3]),targets[3] == targets[4])
    return binary_map