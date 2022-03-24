import numpy as np

def _get_word(n, word_array, length):
    return np.array(word_array[((n - 1) * length):(n * length)])


def encoder(information_words_array):
    
    G = np.array([[1, 0, 0, 0, 1, 1, 1],
                  [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 1]])
    word_length = 4

    codewords = []
    for i in range(1, len(information_words_array)//word_length + 1):
        information_word = _get_word(i, information_words_array, word_length)
        code_word = information_word @ G
        code_word %= 2
        codewords.extend(code_word)

    return np.array(codewords)



