"""Utility Functions"""
from typing import List
import numpy as np
import math
import pickle
import os
import random
from tqdm import tqdm

INPUT_PATH = "./data/input"
OUTPUT_PATH = "./data/output"

def load_input(file:str) -> np.ndarray:
    """Load original file

    Args:
        file (str): name of the original file
    Returns:
        np.ndarray: file data split into 2d matrix of shape (nodes, stripes)
    """
    data_path = f"{INPUT_PATH}/{file}"
    with open(data_path, 'rb') as f:
        return np.array(list(f.read()))

def split_data(data:List, k:int, w:int) -> np.ndarray:
    """split data into stripes

    Args:
        data (List): original data
        k (int): number of data disks
        w (int): size of data block

    Returns:
        ndarray: data splited into blocks in stripes in shape of (stripe, block, data)
    """
    data_size = len(data)
    data_per_stripe = math.ceil(data_size / k)
    block_num = math.ceil(data_per_stripe / w)
    stripe_size = block_num * w
    pad_len = k * stripe_size - data_size
    padded_data = np.pad(data, (0, pad_len), 'constant')
    # split data into stripes
    stripe_data = padded_data.reshape(k, -1)
    return stripe_data

def calc_parity(D:np.ndarray, gf, n:int, m:int) -> np.ndarray:
    """Calculate parity matrix

    Args:
        D (np.ndarray): data from data disks in shape (data_disk number, stripe number)
        gf (galois.GF): GF object to perform galois field arithmetics 
        n (int): number of data disks
        m (int): number of check disks

    Returns:
        np.ndarray: parity matrix in shape (check_disk number, stripe number)
    """
    # calculate Vandermonde matrix
    F = calc_vmatrix(n, m, gf)
    # compute v_matrix * data to obtain parity
    C = np.matmul(gf(F), gf(D))
    return C

def save_to_nodes(D:np.ndarray, C:np.ndarray, shift=True) -> None:
    """save RS codes into nodes

    Args:
        D (np.ndarray): original data block
        C (np.ndarray): parity block
        shift (bool, optional): indicate whether to rotate the stripe. Defaults to True.
    """
    data = np.concatenate((D, C), axis=0)
    if shift:
        data_t = np.transpose(data)
        for i in range(len(data_t)):
            data_t[i] = rotate(data_t[i], i, len(D)+len(C), dir='right')
        data = np.transpose(data_t)
            
    for i in range(len(data)):
        node_path = f"{OUTPUT_PATH}/node_{i}"
        if not os.path.exists(node_path): # create path if not exist
            os.makedirs(node_path)
        file_name = f"{node_path}/data.pickle"
        pickle.dump(data[i], open(file_name, "wb"))
    pass

def rotate(stripe:np.ndarray, stripe_index:int, arr_len:int, dir='right') -> np.ndarray:
    """rotate the elements in the stripe

    Args:
        stripe (np.ndarray): stripe to rotate
        stripe_index (int): row number of the stripe in the data
        arr_len (int): number of total disks of the system
        dir (str, optional): rotation direction (right or left). Defaults to 'right'.

    Returns:
        np.ndarray: rotated stripe
    """
    rotate_num = stripe_index % arr_len
    if dir=='left':
        rotate_num *= -1
    return np.roll(stripe, rotate_num)

def assert_erase_list(erase_list:List, n:int, m:int) -> bool:
    """check if the list of the erased disks is legitimate

    Args:
        erase_list (List): the list of the erased disks
        n (int): the number of data disks
        m (int): the number of parity disks

    Raises:
        Exception: when number of erased disks exceeds the fault tolerance (m)
        Exception: when the disk index in the list is not legitimate

    Returns:
        bool: if the erased disks are legitimate
    """
    try:
        assert len(set(erase_list)) == m
    except:
        raise Exception(f"{len(erase_list)} disks are erased exceeds fault tolerance limit")
    try:
        assert max(erase_list) < n+m and min(erase_list) >= 0
    except:
        raise Exception("disks index out of bound")

def random_corrupt(n:int, m:int)-> List:
    """randomly select disks to be erased

    Args:
        n (int): number of data disks
        m (int): number of parity disks

    Returns:
        List: disk index to be erased
    """
    return random.sample(list(range(n + m)), m)

def clear_disk(erase_list:List) -> None:
    """erase data in target disk

    Args:
        erase_list (List): the list of the erased disks
    """
    for index in erase_list:
        file_path = f"{OUTPUT_PATH}/node_{index}/data.pickle"
        os.remove(file_path)

def restore_data(corrupted_data:np.ndarray, erased_list:List, gf, n:int, m:int, shift=True) -> np.ndarray:
    """restore the original data from the corrupted data

    Args:
        corrupted_data (np.ndarray): data after failures
        erased_list (List): the list of the erased disks
        gf (_type_): Galois algebra handler
        n (int): the number of data disks
        m (int): the number of parity disks
        shift (bool, optional): indicates whether the data is rotated stripe-wise. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    i_matrix = np.identity(n, dtype=int)
    v_matrix = calc_vmatrix(n, m, gf)
    A = np.concatenate((i_matrix, v_matrix), axis=0)

    if shift:
        D = np.zeros((n, corrupted_data.shape[1]),dtype=int)
        c_data_t = np.transpose(corrupted_data)
        for i, stripe in enumerate(tqdm(c_data_t)):
            row_index = [(x-i)%(m+n) for x in range(n+m)]
            A_prime = A[row_index]
            A_prime = np.delete(A_prime, erased_list, axis=0)
            A_prime = gf(A_prime)
            A_inv = np.linalg.inv(A_prime)
            stripe = np.array([stripe])
            D[:,i] = np.transpose(np.matmul(A_inv, gf(np.transpose(stripe))))

    else:
        A = np.delete(A, erased_list, axis=0)
        A = gf(A)
        A_inv = np.linalg.inv(A)
        D = np.matmul(A_inv, gf(corrupted_data))
    return D

def calc_vmatrix(n:int, m:int, gf):
    """calculate the vandermore matrix

    Args:
        n (int): the number of data disks
        m (int): the number of parity disks
        gf (_type_): Galois algebra handler

    Returns:
        _type_: the vandermore matrix
    """
    return [[gf(j+1)**(i) for j in range(n)] for i in range(m)]

def load_nodes():
    """Load and combine data from nodes

    Returns:
        np.ndarray: data loaded from nodes
        List: index of nodes whose data is erased
    """
    data = None
    erase_list = []

    for dir in os.listdir(OUTPUT_PATH):
        file_path = f"{OUTPUT_PATH}/{dir}/data.pickle"
        if os.path.exists(file_path):
            disk_data = pickle.load(open(file_path, 'rb'))
            data = np.array([disk_data]) if data is None else np.append(data, [disk_data], axis=0)
        else:
            disk_num = int(dir.split('_')[1])
            erase_list.append(disk_num)

    return data, erase_list

def verify_correctness(target:str, shift=True) -> None:
    """Check if the resotred data is the same as the original data

    Args:
        target (str): the original file name
        shift (bool, optional): indicates whether the data is rotated stripe-wise. Defaults to True.
    """
    original_data = load_input(target)
    node_data, _ = load_nodes()
    if shift:
        node_data = np.transpose(node_data)
        for i, stripe in enumerate(node_data):
            node_data[i] = rotate(stripe, i, len(node_data), dir='left')
        node_data = np.transpose(node_data)
    
    restored_data = node_data.flatten()[:len(original_data)]
    assert np.array_equal(original_data, restored_data)
    print("Data restoration is correct!")



