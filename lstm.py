# -*-coding: utf-8-*-

import argparse
import os
import numpy as np

#############################
#   class
#############################

class Alphabet(object):
    def __init__(self, chars, chars_are_words):
        '''
        Creates a new alphabet instance from the specified set of characters.
        '''
        if chars_are_words:
            self.chars = '\n'.join(chars)
        else:
            self.chars = ''.join(chars)
            
        self.chars_are_words = chars_are_words
        self.num_chars = len(chars)
        
        self.char_to_index = {}
        self.index_to_char = {}
        
        for i,c in enumerate(chars):
            self.char_to_index[c] = i
            self.index_to_char[i] = c
            
    def char(self,i):
        '''
        Gets the character in the alphabet associated with the specified index.
        '''
        return self.index_to_char[i]
    
    def char_index(self,c):
        '''
        Gets the index in the alphabet associated with the specified character.
        '''
        return self.char_to_index[c]
    
#############################
#   function
#############################
def build_dataset(text, chars_are_words, lookback, stride):
    '''
    Builds a dataset from the specified text with the specified settings by compiling it to a one-hot encoded tensor.
    '''
    if chars_are_words:
        chars = sorted(set(text.split()))
    else:
        chars = sorted(list(set(text)))
        
    alphabet = Alphabet(chars, chars_are_words)
    n = (len(text)-lookback-1)//stride
    x = np.zeros((n, lookback, alphabet.num_chars), dtype=np.bool)
    y = np.zeros((n, alphabet.num_chars), dtype=np.bool)
    
    for i in range(n):
        a = i * stride
        s = text[a:a+lookback]
        yc = text[a+lookback]
        p = i/n
        
        for j,xc in enumerate(s):
            x[i][j][alphabet.char_index[xc]] = 1
        y[i][alphabet.char_index[yc]] = 1
        if (i % 10000) == 0:
            (print (.format "\r{:.2f}%" (* p 100.0)) :end "")))
            print("\r{:.2f}%".format(p*100.0))
    