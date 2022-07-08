import numpy as np

class ConvolutionalBitBlockCoder():
    '''
    '''
    def __init__(self, code_description_dict:dict, decode_value_calculation_method = 'hamming_distance'):
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
        if len(code_description_dict['m_N_previous']) != len(code_description_dict['memory_unit_rules']):
            raise ValueError("Inconsistent size of m_N_previous")


        self.code_rules = code_description_dict
        self.code_rules['number_of_memory_units'] = len(self.code_rules['memory_unit_rules'])
        self.code_rules['number_of_outputs']      = len(self.code_rules['output_rules'])
        self.code_rules['decode_value_calculation_method'] = decode_value_calculation_method
        self.information_word_length = 1
        self.codeword_length = self.code_rules['number_of_outputs']

        # Build the lattice of state transition:
        self.lattice = _create_lattice_skeleton(self.code_rules['number_of_memory_units']);

        for current_state in self.lattice:
            next_state = list(current_state)
            output = self._encode_element(0, next_state)
            self.lattice[current_state][0]['next_state'] = tuple(next_state)
            self.lattice[current_state][0]['output'] = tuple(output)

            next_state = list(current_state)
            output = self._encode_element(1, next_state)
            self.lattice[current_state][1]['next_state'] = tuple(next_state)
            self.lattice[current_state][1]['output'] = tuple(output)
        




    def _encode_element(self, element, m_N_current):
        if element not in {1, 0}:
            raise ValueError("Invalid element")
        if  not isinstance(m_N_current, list): 
            raise TypeError("Invalid type for memory (previous or current) list.")

        m_N_previous = list(m_N_current)
        #Encode the element:
        encoded_output = []
        for rule in self.code_rules['output_rules']:
            encoded_output.append(rule(m_N_current, element) % 2)

        #Update the memory:
        for n, rule in enumerate(self.code_rules['memory_unit_rules']):
            m_N_current[n] = rule(m_N_previous, element) % 2

        return encoded_output
        

    def encode(self, information_words_array):
        if not isinstance(information_words_array, type(np.array([], dtype=np.ubyte))): raise TypeError("The input must be a numpy array.")
        if information_words_array.dtype != np.ubyte: raise TypeError("The input must be a numpy array with element of type np.ubyte")
        
        codewords = []
        m_N_current = list(self.code_rules['m_N_previous'])
        for element in information_words_array:
            codewords.extend(self._encode_element(element, m_N_current))
        return np.array(codewords, dtype=np.ubyte)


    def decode(self, codewords_array, number_of_blocks_per_window = 1000, possible_starting_states = None):
        decoded_message = list() 
        block_size = self.code_rules['number_of_outputs']
        step_size = block_size * number_of_blocks_per_window
        if possible_starting_states is None:
            possible_starting_states = [tuple([0] * self.code_rules['number_of_memory_units'])]
        current_possible_starting_states = possible_starting_states.copy()
        current_decoder_status = _create_current_decoder_status_column(self.code_rules['number_of_memory_units'])
        for i in range(0, len(codewords_array), step_size):
            window = codewords_array[i:(i + step_size)]
            decoded_message.extend(self._decode_window_block(window, current_decoder_status, starting_states = current_possible_starting_states))
            current_possible_starting_states = None

        return np.array(decoded_message, dtype=np.ubyte)

    def _calculate_translation_cost(self, **kwargs):
        cost = 0
        if self.code_rules['decode_value_calculation_method'] == 'hamming_distance':
            received_block = kwargs['received_block']
            potential_block = kwargs['potential_block']
            for element1, element2 in zip(received_block, potential_block):
                if element1 != element2:
                    cost += 1
        elif self.code_rules['decode_value_calculation_method'] == 'gaussian_quadratic_distance':
            received_block = kwargs['received_block']
            potential_block = kwargs['potential_block']
            for element1, element2 in zip(received_block, potential_block):
                if element2 == 0:
                    element2 = -1
                cost += np.power(element1 - element2, 2)
        else:
            raise RuntimeError(f"Method for calculating the node value '{self.code_rules['decode_value_calculation_method']}' is not implemented.")

        return cost


    def _simulate_system_answer_for_element(self, current_state, element, block):
        ''' 
            Simulates the answer of the coding system for the input [element] 
        given the current state [current_state]. It also calculates the 
        [current_value] comparing [block] with output. It returns a 
        tuple with the following structure:
        output: (next_state, output, current_value)
        '''
        next_state = self.lattice[current_state][element]['next_state']
        output = self.lattice[current_state][element]['output']
        kargs = {'received_block':tuple(block), 'potential_block':output}
        current_value = self._calculate_translation_cost(**kargs)
        return next_state, current_value

    def _decode_window_block(self, window:list, current_decoder_status, starting_states = None):
        block_size = self.code_rules['number_of_outputs']
        if len(window) % block_size != 0:
            raise ValueError(f"Invalid window block size. {len(window) =} is not divided by {block_size =}.")
        block_size = self.code_rules['number_of_outputs']
        number_of_blocks_to_process = len(window) / block_size

        # Process the first block: 
        block_index = 0
        if starting_states is not None:
            block = window[block_index:(block_index + block_size)]
            for current_state in starting_states:
                # Simulate next element as 0:
                element = 0
                next_state, current_value = self._simulate_system_answer_for_element(current_state, element, block)
                next_node = current_decoder_status[next_state]['next_node']
                if next_node is None or current_value < next_node.node_cost:
                    current_decoder_status[next_state]['next_node'] = Node(element, current_value, current_decoder_status[current_state]['current_node'])

                # Simulate next element as 1:
                element = 1
                next_state, current_value = self._simulate_system_answer_for_element(current_state, element, block)
                next_node = current_decoder_status[next_state]['next_node']
                if next_node is None or current_value < next_node.node_cost:
                    current_decoder_status[next_state]['next_node'] = Node(element, current_value, current_decoder_status[current_state]['current_node'])
            for current_state in current_decoder_status:
                current_decoder_status[current_state]['current_node'] = current_decoder_status[current_state]['next_node']
                current_decoder_status[current_state]['next_node'] = None
            number_of_blocks_to_process -= 1
            block_index += block_size

        while number_of_blocks_to_process > 0:
            block = window[block_index:(block_index + block_size)]
            for current_state in current_decoder_status:
                # Check for unreachable state:
                if current_decoder_status[current_state]['current_node'] is None:
                    continue

                # Simulate next element as 0:
                element = 0
                next_state, current_value = self._simulate_system_answer_for_element(current_state, element, block)
                next_node = current_decoder_status[next_state]['next_node']
                if next_node is None or current_value < next_node.node_cost:
                    current_decoder_status[next_state]['next_node'] = Node(element, current_value, current_decoder_status[current_state]['current_node'])

                # Simulate next element as 1:
                element = 1
                next_state, current_value = self._simulate_system_answer_for_element(current_state, element, block)
                next_node = current_decoder_status[next_state]['next_node']
                if next_node is None or current_value < next_node.node_cost:
                    current_decoder_status[next_state]['next_node'] = Node(element, current_value, current_decoder_status[current_state]['current_node'])
            for current_state in current_decoder_status:
                current_decoder_status[current_state]['current_node'] = current_decoder_status[current_state]['next_node']
                current_decoder_status[current_state]['next_node'] = None
            number_of_blocks_to_process -= 1
            block_index += block_size

        node_with_smallest_cost = None
        for current_state in current_decoder_status:
            if current_decoder_status[current_state]['current_node'] is None:
                continue
            tmp_node = current_decoder_status[current_state]['current_node']
            current_decoder_status[current_state]['current_node'] = EmptyNode(initial_cost = tmp_node.path_cost)
            if node_with_smallest_cost is None or \
               node_with_smallest_cost.path_cost > tmp_node.path_cost:

                node_with_smallest_cost = tmp_node
        return node_with_smallest_cost.get_sequence()

            



        

class Node():
    def __init__(self, element, additional_cost, next_node):
        if next_node is None:
            path_cost = 0
        else:
            path_cost = next_node.path_cost
        self.path_cost = path_cost + additional_cost
        self.node_cost = additional_cost
        self.next_node = next_node
        self.element = element

    def __repr__(self):
        return f'Node<path_cost:{self.path_cost};node_cost:{self.node_cost};element:{self.element};sequence:"{self.get_sequence()}">'

    def get_sequence(self):
        sequence = list()
        pointer = self
        while pointer.element is not None:
            sequence.append(pointer.element)
            pointer = pointer.next_node
        sequence.reverse()
        return sequence


class EmptyNode(Node):
    def __init__(self, initial_cost = 0):
        super().__init__(None, 0, None)
        self.path_cost = initial_cost

def _create_current_decoder_status_column(number_of_memories:int)->dict:
    ''' 
    This function creates a dict that helps keeping each minimum path to each
    state of the lattice. It is useful during the decoding process. It keeps
    one linked list with its current cost for each possible state or None
    if the state is unreachable.
    '''
    decoder_status_column = dict()
    current_state = [0] * number_of_memories
    for _ in range(2 ** number_of_memories):
        decoder_status_column[tuple(current_state)] = {'current_node':EmptyNode(), 'next_node':None}
        _binary_increment(current_state)
    return decoder_status_column

def _create_lattice_skeleton(number_of_memories:int)->dict:
    lattice_skeleton = dict()
    current_state = [0] * number_of_memories
    for _ in range(2 ** number_of_memories):
        lattice_skeleton[tuple(current_state)] = {1:{'next_state':None, 'output':None}, 0:{'next_state':None, 'output':None}}
        _binary_increment(current_state)
    return lattice_skeleton


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

