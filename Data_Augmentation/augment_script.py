# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:02:45 2022

@author: Lenovo
"""



# This script is used to generate augmented dataset 
# All the augmentation method that may be used in this script can be found in
# https://github.com/boschresearch/data-augmentation-coling2020
# Please see this paper for more detail;

# Xiang Dai and Heike Adel. 2020. An Analysis of Simple Data Augmentation for Named Entity Recognition. In COLING, Online.


import argparse, json, logging, numpy, os, random, sys, torch
from data import ConllCorpus
from train import train, final_test
from augment import get_category2mentions, get_label2tokens
from augment import generate_sentences_by_shuffle_within_segments, generate_sentences_by_replace_mention, generate_sentences_by_replace_token, generate_sentences_by_synonym_replacement



class Augment_data(inputpath)
    corpus = ConllCorpus("development", inputpath, outputpath)
    tag_dict = corpus.build_tag_dict("gold")
    
    print('dataset has been read')
    
    category2mentions = get_category2mentions(corpus.train)
    label2tokens = get_label2tokens(corpus.train, 0)
    f = open(outputpath, 'w', encoding = 'utf-8')
    for s in corpus.train:
        # select one augmentation method that can provide best performance according to their paper
        # set hyperparameters as defaulted values
        aug = generate_sentences_by_synonym_replacement(s, 0.3, 1)[0]
        # cover both original data as well as generated data
        for i in s:
            f.write(str(i) + ' ' + i.get_label('gold') + '\n')
        f.write('\n')
        for i in aug:
            f.write(str(i) + ' ' + i.get_label('gold') + '\n')
        f.write('\n')
    f.close()



