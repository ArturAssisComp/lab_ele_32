import numpy as np
import random

def binary_symmetric_channel(bit_array, p):
    '''
    Description: This function simulates the transmission of bits from
    'bit_array' through a binary symmetric channel. A copy of the original array
    is changed in such a way that each bit has a probability 'p' to change its
    value. The copy is returned.
    Input: bit_array --> numpy array with bits that will be transmitted through
                         the channel.
           p --> parameter that gives the probability of changing each bit. Must be
                 in the range [0, 1].
    Output: numpy array --> new array with some bits changed.
    Time Complexity: O(n)
    Space Complexity: O(n)
    '''
    #Check the value of p:
    if p < 0 or p > 1:
        raise ValueError("p must be in the range [0, 1].")
    if not isinstance(bit_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
    if  bit_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
   

    new_bit_array = bit_array.copy()
    if p != 0:
        random_number_generator = random.SystemRandom()
        for i in range(len(new_bit_array)):
            if random_number_generator.random() <= p:
                new_bit_array[i] ^= 1

    return new_bit_array

def gaussian_channel(codewords_array, mu=0, sigma=1, Eb = 1):
    transmitted_words = list(codewords_array)
    for i in range(len(transmitted_words)):
        random_term = random.gauss(mu=mu, sigma=sigma)
        if transmitted_words[i] == 0:
            transmitted_words[i]  = 0 - np.sqrt(Eb)
        else:
            transmitted_words[i]  = 1 + np.sqrt(Eb)

        transmitted_words[i] += random_term
    return transmitted_words

