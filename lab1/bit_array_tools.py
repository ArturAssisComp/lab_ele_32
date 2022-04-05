import numpy as np
import random

def generate_information_words(num_of_words:int, information_word_length:int):
    """
    Description: This funciton generates a np.array with num_of_words * information_word_length 
    np.ubyte's with random value from {0, 1} each.
    """
    return np.array([random.randint(0, 1) for _ in range(num_of_words * information_word_length)], dtype=np.ubyte)

def compare_arrays(array1, array2):
    """
    Description: This function compares two numpy arrays with the same size. It 
    returns the ration between the number of different element and the total 
    number of elements.
    """
    if not isinstance(array1, type(np.array([]))) or not isinstance(array2, type(np.array([]))):
        raise TypeError("Both arrays must be numpy arrays.")
    if array1.dtype != np.ubyte or array2.dtype != np.ubyte:
        raise TypeError("Both numpy arrays must have dtype of np.ubyte")
    if len(array1) != len(array2):
        raise ValueError("Both arrays must have the same size.")

    return sum(array1 != array2)/len(array1)

