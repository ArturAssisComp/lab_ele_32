import numpy as np

class ConvolutionalBitBlockCoder():
    '''
    '''
    def __init__(self, code_description_dict:dict):
        '''
        Input: 
            code_description_dict --> It is a dictionary that follows the pattern:
                'm_N_previous': (0, 0, ..., 0) --> tuple with initial values that 
                    will be stored into the memories. The len of the tuple is the number 
                    of memories.
                'memory_unit_rules':
                (
                    lambda m_N_previous, input_element: <rules for m0> (ex.: input_element + m_N_previous[0] + m_N_previous[8]),
                    lambda m_N_previous, input_element: <rules for m1>,
                        .
                        .
                        .
                    lambda m_N_previous, input_element: <rules for mi-1>
                )
                'output_rules':
                (
                    lambda m_N_current, input_element: <rules for output0>
                    lambda m_N_current, input_element: <rules for output1>
                        .
                        .
                        .
                    lambda m_N_current, input_element: <rules for outputk-1>
                )
        '''
        #Check the input:
        if not isinstance(code_description_dict, dict):
            raise TypeError("The input must be a dictionary.")
        if code_description_dict.get('m_N_previous', None) is None or\
           code_description_dict.get('memory_unit_rules', None) is None or\
           code_description_dict.get('output_rules', None) is None:
            raise ValueError("Incomplete input dictionary.")
        if not isinstance(code_description_dict['m_N_previous'], tuple) or\
           not isinstance(code_description_dict['memory_unit_rules'], tuple) or\
           not isinstance(code_description_dict['output_rules'], tuple):
            raise ValueError("Incorrect value for key in dict.")


        self.code_rules = code_description_dict
        self.code_rules['number_of_memory_units'] = len(self.code_rules['memory_unit_rules'])
        self.code_rules['number_of_outputs']      = len(self.code_rules['output_rules'])



    def encode_element(self, element, m_N_previous, m_N_current):
        if element not in {1, 0}:
            raise ValueError("Invalid element")
        if not isinstance(m_N_previous, list)  or\
           not isinstance(m_N_current, list): 
            raise TypeError("Invalid type for memory (previous or current) list.")
        if len(m_N_previous) != len(m_N_current):
            raise ValueError("Inconsistent sizes for memory lists.")

        #Encode the element:
        encoded_output = []
        for rule in self.code_rules['output_rules']:
            encoded_output.append(rule(m_N_current, element) % 2)

        #Update the memory:
        for i in range(len(m_N_previous)):
            m_N_previous[i] = m_N_current[i]
        
        for n, rule in enumerate(self.code_rules['memory_unit_rules']):
            m_N_current[n] = rule(m_N_previous, element) % 2

        return encoded_output
        

    def encode(self, information_words_array):
        if not isinstance(information_words_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if information_words_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
        
        codewords = []
        m_N_current = list(self.code_rules['m_N_previous'])
        m_N_previous = [None] * self.code_rules['number_of_memory_units']
        for element in information_words_array:
            codewords.extend(self.encode_element(element, m_N_previous, m_N_current))
        return np.array(codewords, dtype=np.ubyte)


    def decode(self, codewords_array):
        pass






###############################################################################
#                                  TEST CASE                                  #
###############################################################################
import unittest
'''
Class name: ConvolutionalBitBlockCoder
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
            ConvolutionalBitBlockCoder('hello')

    def test_CBBC__I__002(self):
        pass

    def test_CBBC__I__003(self):
        pass





if __name__=='__main__':
    unittest.main(argv=[''], verbosity=3,exit=False)

