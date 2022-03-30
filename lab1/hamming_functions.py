import numpy as np

def _get_word(n:int, word_array, length:int):
    '''
    Description: This function returns the n-th slice of length 'length' from word_array.
    Input: n          --> Positive integer {1, 2, 3, ...} that indicates the position of the word.
           length     --> The length of the word.
           word_array --> sequence of items

    Output: sequence with maximum length equals to 'length'.
    '''
    if not isinstance(n, int): raise TypeError("The input 'n' (first input) must be an integer")
    if not isinstance(length, int): raise TypeError("The input 'length' (third input) must be an integer")
    return word_array[((n - 1) * length):(n * length)]


def hamming_encoder(information_words_array):
    '''
    Function ID: HE
    Description: This function encodes the information words from 
    'information_words_array'. The encoding process is the Hamming code.

    Information word -> b1, b2, b3, b4
    Codeword         -> b1, b2, b3, b4, p1, p2, p3

    Input: (np.array) information_words_array --> numpy array with length that 
                                                  is multiple of 4 (the size of 
                                                  each word. Each element is 
                                                  either a 0 or a 1.
    Output: (np.array) --> numpy array of np.ubyte with len(information_words_array)/4 codewords.
                           Each codeword has 7 elements.
    '''
    word_length = 4
    if not isinstance(information_words_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
    if information_words_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
    if len(information_words_array) % word_length != 0: raise ValueError(f"The input must have a len that is multiple of {word_length}")
    
    # The matrix G is used to encode the word using hamming code.
    G = np.array([[1, 0, 0, 0, 1, 1, 1],  # b1
                  [0, 1, 0, 0, 1, 0, 1],  # b2
                  [0, 0, 1, 0, 1, 1, 0],  # b3
                  [0, 0, 0, 1, 0, 1, 1]]) # b4
                 # b1 b2 b3 b4 p1 p2 p3

    codewords = []
    for ini in range(0, len(information_words_array), word_length):
        end = ini + word_length
        information_word = information_words_array[ini: end]
        code_word = information_word @ G
        code_word %= 2
        codewords.extend(code_word)

    return np.array(codewords, dtype=np.ubyte)


def hamming_decoder(codewords_array):
    '''
    Function ID: HD
    Description: This function recovers a candidate for information word from each 
    codeword in codewords_array and returns those candidates in an array with the 
    same order they were produced. The information word returned is not necessarily
    wqual to the original information word, it is only a guess based on Hamming code.

    Codeword                   -> b1, b2, b3, b4, p1, p2, p3
    Guess for information word -> b1', b2', b3', b4'

    Input: (np.array) codewords_array --> numpy array with length that 
                                                  is multiple of 7 (the size of 
                                                  each codeword. Each element is 
                                                  either a 0 or a 1.
    Output: (np.array) --> numpy array of np.ubyte with len(codewords_array)/7 
                           candidates of information words.Each information word 
                           has 4 elements.
    '''
    codeword_length = 7
    word_length = 4
    if not isinstance(codewords_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
    if codewords_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
    if len(codewords_array) % codeword_length != 0: raise ValueError(f"The input must have a len that is multiple of {codeword_length}")
    
    # Matriz de verificacao de paridade
    H_T = [[1, 1, 1], # b1
           [1, 0, 1], # b2
           [1, 1, 0], # b3
           [0, 1, 1], # b4
           [1, 0, 0], # p1
           [0, 1, 0], # p2
           [0, 0, 1]] # p3
          # s1 s2 s3

    candidate_information_words = []
    for ini in range(0, len(codewords_array), codeword_length):
        end = ini + codeword_length
        received_codeword = codewords_array[ini : end]
        syndrome = received_codeword @ H_T
        syndrome %= 2
        candidate_for_error = _get_min_hamming_weight_error_for_given_syndrome(syndrome)
        candidate_codeword = np.bitwise_xor(candidate_for_error, received_codeword)
        candidate_information_words.extend(candidate_codeword[0 : word_length])
    return np.array(candidate_information_words, dtype=np.ubyte)
    

def _get_min_hamming_weight_error_for_given_syndrome(syndrome):
    # We have few possibilities (2**3 = 8), thus, they are going to be hardcoded.

    #   Syndromes         Error with min Hamming weight
    #   (s1, s2, s3) -->  [b1, b2, b3, b4, p1, p2, p3]
    #   ( 0,  0,  0) -->  [ 0,  0,  0,  0,  0,  0,  0]
    #   ( 0,  0,  1) -->  [ 0,  0,  0,  0,  0,  0,  1]
    #   ( 0,  1,  0) -->  [ 0,  0,  0,  0,  0,  1,  0]
    #   ( 0,  1,  1) -->  [ 0,  0,  0,  1,  0,  0,  0]
    #   ( 1,  0,  0) -->  [ 0,  0,  0,  0,  1,  0,  0]
    #   ( 1,  0,  1) -->  [ 0,  1,  0,  0,  0,  0,  0]
    #   ( 1,  1,  0) -->  [ 0,  0,  1,  0,  0,  0,  0]
    #   ( 1,  1,  1) -->  [ 1,  0,  0,  0,  0,  0,  0]
    if not hasattr(_get_min_hamming_weight_error_for_given_syndrome, "syn_to_error_dict"):
        _get_min_hamming_weight_error_for_given_syndrome.syn_to_error_dict = {
                (0,  0,  0) : [ 0,  0,  0,  0,  0,  0,  0],
                (0,  0,  1) : [ 0,  0,  0,  0,  0,  0,  1],
                (0,  1,  0) : [ 0,  0,  0,  0,  0,  1,  0],
                (0,  1,  1) : [ 0,  0,  0,  1,  0,  0,  0],
                (1,  0,  0) : [ 0,  0,  0,  0,  1,  0,  0],
                (1,  0,  1) : [ 0,  1,  0,  0,  0,  0,  0],
                (1,  1,  0) : [ 0,  0,  1,  0,  0,  0,  0],
                (1,  1,  1) : [ 1,  0,  0,  0,  0,  0,  0]
                }

    return _get_min_hamming_weight_error_for_given_syndrome.syn_to_error_dict[tuple(syndrome)]
       
    
    



###############################################################################
#                                  TEST CASE                                  #
###############################################################################
import unittest
'''
Function name: hamming_encoder
ID: HE
'''
class TestHE(unittest.TestCase):
    '''
    Test cases:
    HE001 - Test with wrong inputs.

    HE002 - Test some edge cases.
    
    HE003 - Test cases for each partition.
    '''
    def test_HE001(self):
        with self.assertRaises(TypeError):
            hamming_encoder([1, 2, 3])
        with self.assertRaises(TypeError):
            hamming_encoder(np.array([1, 2, 3, 5], dtype=int))
        with self.assertRaises(ValueError):
            hamming_encoder(np.array([1, 2, 3], dtype=np.ubyte))
        with self.assertRaises(TypeError):
            hamming_encoder(123)
        with self.assertRaises(TypeError):
            hamming_encoder([])


    def test_HE002(self):
        #No information word:
        information_words = np.array([], dtype=np.ubyte)
        codewords_expected = np.array([], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

        # 1 information word
        ## only 0s
        information_words = np.array([0, 0, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

        ## only 1s
        information_words = np.array([1, 1, 1, 1], dtype=np.ubyte)
        codewords_expected = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

        ## 3 1s
        information_words = np.array([1, 0, 1, 1], dtype=np.ubyte)
        codewords_expected = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

        # 2 information word
        ## only 0s
        information_words = np.array([0, 0, 0, 0] + [0, 0, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

        ## only 1s in the first word and 0s in the second.
        information_words = np.array([1, 1, 1, 1] + [0, 0, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([1, 1, 1, 1, 1, 1, 1] + [0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

    def test_HE003(self):
        # 2 information word
        information_words = np.array([1, 0, 1, 0] + [0, 1, 1, 1], dtype=np.ubyte)
        codewords_expected = np.array([1, 0, 1, 0, 0, 0, 1] + [0, 1, 1, 1, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())

        # 3 information word
        information_words = np.array([0, 1, 0, 0] + [1, 1, 0, 1] + [1, 1, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([0, 1, 0, 0, 1, 0, 1] + [1, 1, 0, 1, 0, 0, 1] + [1, 1, 0, 0, 0, 1, 0], dtype=np.ubyte)
        self.assertEqual(hamming_encoder(information_words).tolist(), codewords_expected.tolist())


'''
Function name: hamming_decoder 
ID: HD 
'''
class TestHD(unittest.TestCase):
    '''
    Test cases:
    HD001 - Test with wrong inputs.

    HD002 - Test some edge cases.
    
    HD003 - Test cases for each partition.
    '''
    def test_HD001(self):
        with self.assertRaises(TypeError):
            hamming_decoder([1, 2, 3])
        with self.assertRaises(TypeError):
            hamming_decoder(np.array([1, 2, 3, 5, 6, 7], dtype=int))
        with self.assertRaises(TypeError):
            hamming_decoder(123)
        with self.assertRaises(TypeError):
            hamming_decoder([])
        with self.assertRaises(ValueError):
            hamming_decoder(np.array([1, 2, 3], dtype=np.ubyte))


    def test_HD002(self):
        #No code word:
        codewords = np.array([], dtype=np.ubyte)
        information_words_expected = np.array([], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        # 1 code word
        ## only 0s
        codewords = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        ## p1 is wrong:
        codewords = np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        ## p2 is wrong:
        codewords = np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        ## b3 wrong:
        codewords = np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        ## b1 wrong:
        codewords = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        ## b1 wrong:
        codewords = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.ubyte)
        information_words_expected = np.array([1, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        ## b2 wrong:
        codewords = np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.ubyte)
        information_words_expected = np.array([1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())

        # 2 Code words:
        ## b1 wrong for the first codeword and b2 wrong for the second:
        codewords = np.array([0, 0, 0, 0, 1, 1, 1] + [1, 0, 1, 1, 1, 1, 1], dtype=np.ubyte)
        information_words_expected = np.array([1, 0, 0, 0] + [1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())




    def test_HD003(self):
        # 3 codewords:
        codewords = np.array([0, 1, 1, 0, 1, 0, 1] + [0, 0, 0, 1, 0, 1, 1] + [1, 1, 1, 1, 1, 1, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 1, 0, 0] + [0, 0, 0, 1] + [1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(hamming_decoder(codewords).tolist(), information_words_expected.tolist())



if __name__=='__main__':
    unittest.main(argv=[''], verbosity=3,exit=False)

