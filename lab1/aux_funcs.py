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
    if len(information_words_array) % word_length != 0: raise ValueError(f"The input must have a len that is multiple of {word_length}")
    
    # The matrix G is used to encode the word using hamming code.
    G = np.array([[1, 0, 0, 0, 1, 1, 1],
                  [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 1]])

    codewords = []
    for ini in range(0, len(information_words_array), word_length):
        end = ini + word_length
        information_word = information_words_array[ini: end]
        code_word = information_word @ G
        code_word %= 2
        codewords.extend(code_word)

    return np.array(codewords, dtype=np.ubyte)



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
        with self.assertRaises(ValueError):
            hamming_encoder(np.array([1, 2, 3]))
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
Function name: abc
ID: ABC
'''
class TestS(unittest.TestCase):
    '''
    Test cases:
    S001 - Test with wrong inputs.

    S002 - Test some edge cases.
    
    S003 - Test cases for each partition.
    '''
    def test_S001(self):
        self.assertEqual(2, 2)


    def test_S002(self):
        pass


    def test_S003(self):
        pass


if __name__=='__main__':
    unittest.main(argv=[''], verbosity=3,exit=False)

