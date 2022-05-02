import numpy as np
from numpy.polynomial import Polynomial as P
class CyclicBitBlockCoder():
    '''
    Description: This class is an implementation of a cyclic block coder. It is 
    initialized passing its generator polynomial (g) and the size of the codeword
    (N). 
    Constraints: 
    - len(g) < N;
    - g(D) is a factor of the polynomial (1 + D^N);
    - g[0] == 1;
    - g[-1] == 1;


    k = N - len(g)
    k -> the number of information bits
    N -> the number of bits of a codeword.
    
    Transmission rate -> k / N

    Ex.: To create Hamming code coder, use the following matrix:
    | 1  0  0  0  1  1  1 | b1
    | 0  1  0  0  1  0  1 | b2
    | 0  0  1  0  1  1  0 | b3
    | 0  0  0  1  0  1  1 | b4
     b1 b2 b3 b4 p1 p2 p3
    '''
    def __init__(self, g, N):
        '''
        Inputs: 
            g -> The generator polynomial must be a list with the coefficients represented
                 as: g(D) = a0 + a1*D + ... + a(N-k)*D^(N-k) -> [a0, a1, ..., a(N-k)]
                 ai must be 0 or 1 for any i and a0 = a(N-k) = 1
        '''
        if not isinstance(g, list):
            raise TypeError("The generator polynomial (first argument) must be a list.")
        if not isinstance(N, int) or N <= 0:
            raise TypeError("The size of the codeword (second argument) must be a positive int.")
        if not set(g).issubset({1, 0}):
            raise ValueError("The generator polynomial must have only 0s and 1s as coefficients.")
        if g[0] != 1 or g[-1] != 1:
            raise ValueError("The generator polynomial must have the last and the first coefficients equal 1.")
        if len(g) >= N + 1: 
            raise ValueError(f"The length of the generator polynomial ({len(g)}) must be less than N + 1 ({N + 1}).")

        self.generator_polynomial = P(g)
        self.codeword_length = N 
        self.information_word_length = N - len(g) + 1
        self._init_min_weight_error_for_given_syndrome()


    def _init_min_weight_error_for_given_syndrome(self):
        pass
        '''
        _, syndrome_length = self.parity_check_matrix.shape
        self.min_weight_error_for_given_syndrome = _generate_dict_with_default_syndromes(syndrome_length, default_value=(None,self.codeword_length+1))

        #Possible errors:
        current_error = [0] * self.codeword_length
        for _ in range(2 ** self.codeword_length):
            syndrome = tuple((np.array(current_error) @ self.parity_check_matrix) % 2)
            weight = sum(current_error)
            if weight <= self.min_weight_error_for_given_syndrome[syndrome][1]: # The choose of the comparison operator makes difference here. <= instead of <?
                #Update the error for the current syndrome:
                self.min_weight_error_for_given_syndrome[syndrome] = (tuple(current_error), weight)
            _binary_increment(current_error)
        '''
        

    def encode(self, information_words_array):
        word_length = self.information_word_length
        if not isinstance(information_words_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if information_words_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
        if len(information_words_array) % word_length != 0: raise ValueError(f"The input must have a len that is multiple of {word_length}")
        
        codewords = []
        for ini in range(0, len(information_words_array), word_length):
            end = ini + word_length
            information_word = information_words_array[ini: end]
            code_word = (self.generator_polynomial * P(information_word)).coef % 2
            codewords.extend(code_word.tolist() + [0]*(self.codeword_length - len(code_word)))
        return np.array(codewords, dtype=np.ubyte)


    def decode(self, codewords_array):
        '''
        codeword_length = self.codeword_length
        word_length = self.information_word_length
        default_error = [0] * codeword_length
        if not isinstance(codewords_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if  codewords_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
        if len(codewords_array) % codeword_length != 0: raise ValueError(f"The input must have a len that is multiple of {codeword_length}")
        
        # Matriz de verificacao de paridade
        H_T = self.parity_check_matrix

        candidate_information_words = []
        for ini in range(0, len(codewords_array), codeword_length):
            end = ini + codeword_length
            received_codeword = codewords_array[ini : end]
            syndrome = received_codeword @ H_T
            syndrome %= 2
            syndrome = tuple(syndrome)
            candidate_for_error, _ = self.min_weight_error_for_given_syndrome[syndrome] 
            if candidate_for_error is None: canditate_for_error = default_error
            candidate_codeword = np.bitwise_xor(candidate_for_error, received_codeword)
            candidate_information_words.extend(candidate_codeword[0 : word_length])
        return np.array(candidate_information_words, dtype=np.ubyte)
        '''
        pass


def _generate_dict_with_default_syndromes(syndrome_length, default_value=None):
    """
    Description: Creates a dictionary with 2**syndrome_length pairs of key:value. 
    The keys are tuples that represent binary values from (0, 0, .., 0) to 
    (1, 1, ..., 1) with length equals to 'syndrome_length'. All of those pairs 
    have value equals to 'default_value'.
    T = O(2**syndrome_length)
    """
    '''
    current_syndrome = [0] * syndrome_length
    result_dict = dict()

    for _ in range(2**syndrome_length):
        result_dict[tuple(current_syndrome)] = default_value
        _binary_increment(current_syndrome)

    return result_dict
    '''
    pass



###############################################################################
#                                  TEST CASE                                  #
###############################################################################
import unittest
'''
Class name: CyclicBitBlockCoder
Method name: __init__
ID: CBBC__I__
'''
class TestCBBC__I__(unittest.TestCase):
    '''
    Test cases:
    CBBC__I__001 - Test with wrong inputs.

    CBBC__I__002 - Test some edge cases.
    
    CBBC__I__003 - Test cases for each partition.
    '''
    def test_CBBC__I__001(self):
        with self.assertRaises(TypeError):
            CyclicBitBlockCoder('hello', 12)
        with self.assertRaises(TypeError):
            CyclicBitBlockCoder([], '12')
        with self.assertRaises(ValueError):
            CyclicBitBlockCoder([1, 2, 3], 12)
        with self.assertRaises(ValueError):
            CyclicBitBlockCoder([1, 2, 1], 12)
        with self.assertRaises(ValueError):
            CyclicBitBlockCoder([1, 0, 1], 2)
        with self.assertRaises(ValueError):
            CyclicBitBlockCoder([1, 0, 1], 3)
        with self.assertRaises(ValueError):
            CyclicBitBlockCoder([0, 0, 1], 4)

    def test_CBBC__I__002(self):
        pass

    def test_CBBC__I__003(self):
        pass
        '''
        hamming_encode_matrix = np.array([[1, 0, 0, 0, 1, 1, 1],  # b1
                                          [0, 1, 0, 0, 1, 0, 1],  # b2
                                          [0, 0, 1, 0, 1, 1, 0],  # b3
                                          [0, 0, 0, 1, 0, 1, 1]]) # b4
                                         # b1 b2 b3 b4 p1 p2 p3

        hamming_coder = GenericParityBitBlockCoder(hamming_encode_matrix)
        self.assertEqual(hamming_coder.information_word_length, 4)
        self.assertEqual(hamming_coder.codeword_length, 7)
        self.assertEqual(hamming_coder.encode_matrix.tolist(), [[1, 0, 0, 0, 1, 1, 1],  
                                                               [0, 1, 0, 0, 1, 0, 1],  
                                                               [0, 0, 1, 0, 1, 1, 0], 
                                                               [0, 0, 0, 1, 0, 1, 1]])
        self.assertEqual(hamming_coder.parity_check_matrix.tolist(), [[1, 1, 1], 
                                                                      [1, 0, 1], 
                                                                      [1, 1, 0], 
                                                                      [0, 1, 1], 
                                                                      [1, 0, 0], 
                                                                      [0, 1, 0], 
                                                                      [0, 0, 1]])
        '''





if __name__=='__main__':
    unittest.main(argv=[''], verbosity=3,exit=False)

