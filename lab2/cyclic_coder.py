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

    '''
    def __init__(self, g, N, num_of_bits_to_correct=1):
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
        self._init_min_weight_error_for_given_syndrome(num_of_bits_to_correct)


    def _init_min_weight_error_for_given_syndrome(self, num_of_bits_to_correct):
        syndrome_length = self.codeword_length - self.information_word_length

        #Correct last bit:
        last_bit_error_coef = tuple([0] * (self.codeword_length - 1) + [1])
        last_bit_error_polynomial = P(last_bit_error_coef)
        syndrome_coef = tuple((last_bit_error_polynomial % self.generator_polynomial).coef % 2)
        self.min_weight_error_for_given_syndrome = dict()
        self.min_weight_error_for_given_syndrome[syndrome_coef] = (last_bit_error_coef, 1) 

        #Implement correction for more bits:
        # ...
        

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
        codeword_length = self.codeword_length
        word_length = self.information_word_length
        if not isinstance(codewords_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if  codewords_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
        if len(codewords_array) % codeword_length != 0: raise ValueError(f"The input must have a len that is multiple of {codeword_length}")
        

        candidate_information_words = []
        for ini in range(0, len(codewords_array), codeword_length):
            end = ini + codeword_length
            received_codeword = list(codewords_array[ini : end])
            received_codeword_polynomial = P(received_codeword)
            #Meggit decoding algorithm:
            for _ in range(self.codeword_length):
                syndrome = tuple((received_codeword_polynomial % self.generator_polynomial).coef % 2)
                if self.min_weight_error_for_given_syndrome.get(syndrome)  is not None:
                    received_codeword[-1] ^= 1 #fix the last bit

                #rotate the codeword:
                received_codeword = received_codeword[-1:] + received_codeword[:-1]
                received_codeword_polynomial = P(received_codeword)
            candidate_information_word = (received_codeword_polynomial // self.generator_polynomial).coef % 2
            candidate_information_word = list(candidate_information_word) + [0]*(word_length - len(candidate_information_word))
            candidate_information_words.extend(candidate_information_word)
        return np.array(candidate_information_words, dtype=np.ubyte)




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





if __name__=='__main__':
    unittest.main(argv=[''], verbosity=3,exit=False)

