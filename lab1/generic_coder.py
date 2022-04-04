import numpy as np

class GenericParityBitBlockCoder():
    '''
    Description: This class is an implementation of a generic block coder that 
    uses parity bits. It is initialized using its encoding matrix. The encoding 
    matrix is as follows:


    # The matrix that will be used to encode the information word to codeword.
                  |A11 A12 ...             A1l| b1
                  |A21 ...                    | b2
                  |.                          |  .
                  |.                          |  .
                  |.                          |  .
                  |Ak1                     Akl| bk
                    b1  b2 ... bk p1 p2 ... pl
    Constraints: 
    Aij in {0, 1} for any i, and j
    Aii == 1
    Aji == 0 if j != i and i <= k

    Definitions:
    k -> the number of information bits
    l -> the number of parity bits
    
    Transmission rate -> k / (l + k)

    bi, with  1 <= i <= k -> information bit
    pj, with  1 <= j <= l -> parity bit. Each parity bit will be added (XOR) to 
    a group of information bits and the result must be 0. Thus, the parity bit 
    will be chosen in such a way that the result is 0.

    Ex.: To create Hamming code coder, use the following matrix:
    | 1  0  0  0  1  1  1 | b1
    | 0  1  0  0  1  0  1 | b2
    | 0  0  1  0  1  1  0 | b3
    | 0  0  0  1  0  1  1 | b4
     b1 b2 b3 b4 p1 p2 p3
    '''
    def __init__(self, encode_matrix):
        '''
        Obs.:
        The matrix |1, 0| is [[1, 0]] not [1, 0]
        '''
        self.encode_matrix = np.array(encode_matrix)
        try:
            self.information_word_length, self.codeword_length = self.encode_matrix.shape
        except ValueError:
            raise TypeError("The input is not an array.")


        if not np.logical_or(self.encode_matrix == 1, self.encode_matrix == 0).all():
            raise ValueError("The encode matrix may have only 0s and 1s.")
        if self.information_word_length > self.codeword_length:
            raise ValueError("Invalid encoding matrix. The number of bits in codeword cannot be smaller than the number of information bits.")

        information_part_of_encode_matrix = self.encode_matrix[:self.information_word_length, :self.information_word_length]
        comparison = np.identity(self.information_word_length) != information_part_of_encode_matrix
        if comparison.any():
            raise ValueError(f"Invalid encoding matrix. The First part of the matrix must be the identity with size {self.information_word_length}.")

        self.parity_check_matrix = np.concatenate([self.encode_matrix[:, self.information_word_length:], np.identity(self.codeword_length - self.information_word_length)], axis=0)

        self._init_min_weight_error_for_given_syndrome()


    def _init_min_weight_error_for_given_syndrome(self):
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
        

    def encode(self, information_words_array):
        word_length = self.information_word_length
        if not isinstance(information_words_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if information_words_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
        if len(information_words_array) % word_length != 0: raise ValueError(f"The input must have a len that is multiple of {word_length}")
        
        # The matrix G is used to encode the word using hamming code.
        G = self.encode_matrix
        codewords = []
        for ini in range(0, len(information_words_array), word_length):
            end = ini + word_length
            information_word = information_words_array[ini: end]
            code_word = information_word @ G
            code_word %= 2
            codewords.extend(code_word)
        return np.array(codewords, dtype=np.ubyte)


    def decode(self, codewords_array):
        codeword_length = self.codeword_length
        word_length = self.information_word_length
        default_error = [0] * codeword_length
        if not isinstance(codewords_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if codewords_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
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


def _generate_dict_with_default_syndromes(syndrome_length, default_value=None):
    """
    Description: Creates a dictionary with 2**syndrome_length pairs of key:value. 
    The keys are tuples that represent binary values from (0, 0, .., 0) to 
    (1, 1, ..., 1) with length equals to 'syndrome_length'. All of those pairs 
    have value equals to 'default_value'.
    T = O(2**syndrome_length)
    """
    current_syndrome = [0] * syndrome_length
    result_dict = dict()

    for _ in range(2**syndrome_length):
        result_dict[tuple(current_syndrome)] = default_value
        _binary_increment(current_syndrome)

    return result_dict

def _binary_increment(binary_number:list)->None:
    carry = 1
    for i in range(len(binary_number)):
        current_index = len(binary_number) - 1 - i
        current_bit = binary_number[current_index]
        carry, binary_number[current_index] = _add_bit(carry, current_bit)
        if carry == 0: break

def _add_bit(bit1, bit2):
    carry = bit1 & bit2
    result = bit1 ^ bit2
    return (carry, result)


###############################################################################
#                                  TEST CASE                                  #
###############################################################################
import unittest
'''
Class name: GenericParityBitBlockCoder
Method name: __init__
ID: GPBB__I__
'''
class TestGPBB__I__(unittest.TestCase):
    '''
    Test cases:
    GPBB__I__001 - Test with wrong inputs.

    GPBB__I__002 - Test some edge cases.
    
    GPBB__I__003 - Test cases for each partition.
    '''
    def test_GPBB__I__001(self):
        with self.assertRaises(TypeError):
            GenericParityBitBlockCoder([1, 2, 3])
        with self.assertRaises(TypeError):
            GenericParityBitBlockCoder(123)
        with self.assertRaises(TypeError):
            GenericParityBitBlockCoder()
        with self.assertRaises(ValueError):
            GenericParityBitBlockCoder([[]])
        with self.assertRaises(ValueError):
            GenericParityBitBlockCoder([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]])
        with self.assertRaises(ValueError):
            GenericParityBitBlockCoder([[1, 2, 3]])
        with self.assertRaises(ValueError):
            GenericParityBitBlockCoder([[0]])


    def test_GPBB__I__002(self):
        simple_bit = GenericParityBitBlockCoder([[1]])
        self.assertEqual(simple_bit.information_word_length, 1)
        self.assertEqual(simple_bit.codeword_length, 1)
        self.assertEqual(simple_bit.encode_matrix.tolist(), [[1]])
        self.assertEqual(simple_bit.parity_check_matrix.tolist(), [[]])

        two_bits_1_info_bit = GenericParityBitBlockCoder([[1, 1]])
        self.assertEqual(two_bits_1_info_bit.information_word_length, 1)
        self.assertEqual(two_bits_1_info_bit.codeword_length, 2)
        self.assertEqual(two_bits_1_info_bit.encode_matrix.tolist(), [[1, 1]])
        self.assertEqual(two_bits_1_info_bit.parity_check_matrix.tolist(), [[1], [1]])

    def test_GPBB__I__003(self):
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
Class name: GenericParityBitBlockCoder
Method name: encode
ID: GPBBE
'''
class TestGPBBE(unittest.TestCase):
    '''
    Test cases:
    GPBBE001 - Test with wrong inputs.

    GPBBE002 - Test some edge cases.
    
    GPBBE003 - Test cases for each partition.
    '''
    def setUp(self):
        hamming_encode_matrix = np.array([[1, 0, 0, 0, 1, 1, 1],  # b1
                                          [0, 1, 0, 0, 1, 0, 1],  # b2
                                          [0, 0, 1, 0, 1, 1, 0],  # b3
                                          [0, 0, 0, 1, 0, 1, 1]]) # b4
                                         # b1 b2 b3 b4 p1 p2 p3

        self.hamming_coder = GenericParityBitBlockCoder(hamming_encode_matrix)

    def test_GPBBE001(self):
        with self.assertRaises(TypeError):
            self.hamming_coder.encode([1, 2, 3])
        with self.assertRaises(TypeError):
            self.hamming_coder.encode(np.array([1, 2, 3, 5], dtype=int))
        with self.assertRaises(ValueError):
            self.hamming_coder.encode(np.array([1, 2, 3], dtype=np.ubyte))
        with self.assertRaises(TypeError):
            self.hamming_coder.encode(123)
        with self.assertRaises(TypeError):
            self.hamming_coder.encode([])


    def test_GPBBE002(self):
        #No information word:
        information_words = np.array([], dtype=np.ubyte)
        codewords_expected = np.array([], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

        # 1 information word
        ## only 0s
        information_words = np.array([0, 0, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

        ## only 1s
        information_words = np.array([1, 1, 1, 1], dtype=np.ubyte)
        codewords_expected = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

        ## 3 1s
        information_words = np.array([1, 0, 1, 1], dtype=np.ubyte)
        codewords_expected = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

        # 2 information word
        ## only 0s
        information_words = np.array([0, 0, 0, 0] + [0, 0, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

        ## only 1s in the first word and 0s in the second.
        information_words = np.array([1, 1, 1, 1] + [0, 0, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([1, 1, 1, 1, 1, 1, 1] + [0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

    def test_GPBBE003(self):
        # 2 information word
        information_words = np.array([1, 0, 1, 0] + [0, 1, 1, 1], dtype=np.ubyte)
        codewords_expected = np.array([1, 0, 1, 0, 0, 0, 1] + [0, 1, 1, 1, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())

        # 3 information word
        information_words = np.array([0, 1, 0, 0] + [1, 1, 0, 1] + [1, 1, 0, 0], dtype=np.ubyte)
        codewords_expected = np.array([0, 1, 0, 0, 1, 0, 1] + [1, 1, 0, 1, 0, 0, 1] + [1, 1, 0, 0, 0, 1, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.encode(information_words).tolist(), codewords_expected.tolist())


'''
Class name: GenericParityBitBlockCoder
Method name: decode
ID: GPBBD
'''
class TestGPBBD(unittest.TestCase):
    '''
    Test cases:
    GPBBD001 - Test with wrong inputs.

    GPBBD002 - Test some edge cases.
    
    GPBBD003 - Test cases for each partition.
    '''

    def setUp(self):
        hamming_encode_matrix = np.array([[1, 0, 0, 0, 1, 1, 1],  # b1
                                          [0, 1, 0, 0, 1, 0, 1],  # b2
                                          [0, 0, 1, 0, 1, 1, 0],  # b3
                                          [0, 0, 0, 1, 0, 1, 1]]) # b4
                                         # b1 b2 b3 b4 p1 p2 p3

        self.hamming_coder = GenericParityBitBlockCoder(hamming_encode_matrix)

    def test_GPBBD001(self):
        with self.assertRaises(TypeError):
            self.hamming_coder.decode([1, 2, 3])
        with self.assertRaises(TypeError):
            self.hamming_coder.decode(np.array([1, 2, 3, 5, 6, 7], dtype=int))
        with self.assertRaises(TypeError):
            self.hamming_coder.decode(123)
        with self.assertRaises(TypeError):
            self.hamming_coder.decode([])
        with self.assertRaises(ValueError):
            self.hamming_coder.decode(np.array([1, 2, 3], dtype=np.ubyte))


    def test_GPBBD002(self):
        #No code word:
        codewords = np.array([], dtype=np.ubyte)
        information_words_expected = np.array([], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        # 1 code word
        ## only 0s
        codewords = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        ## p1 is wrong:
        codewords = np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        ## p2 is wrong:
        codewords = np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        ## b3 wrong:
        codewords = np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        ## b1 wrong:
        codewords = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        ## b1 wrong:
        codewords = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.ubyte)
        information_words_expected = np.array([1, 0, 0, 0], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        ## b2 wrong:
        codewords = np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.ubyte)
        information_words_expected = np.array([1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())

        # 2 Code words:
        ## b1 wrong for the first codeword and b2 wrong for the second:
        codewords = np.array([0, 0, 0, 0, 1, 1, 1] + [1, 0, 1, 1, 1, 1, 1], dtype=np.ubyte)
        information_words_expected = np.array([1, 0, 0, 0] + [1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())




    def test_GPBBD003(self):
        # 3 codewords:
        codewords = np.array([0, 1, 1, 0, 1, 0, 1] + [0, 0, 0, 1, 0, 1, 1] + [1, 1, 1, 1, 1, 1, 0], dtype=np.ubyte)
        information_words_expected = np.array([0, 1, 0, 0] + [0, 0, 0, 1] + [1, 1, 1, 1], dtype=np.ubyte)
        self.assertEqual(self.hamming_coder.decode(codewords).tolist(), information_words_expected.tolist())



if __name__=='__main__':
    unittest.main(argv=[''], verbosity=3,exit=False)

