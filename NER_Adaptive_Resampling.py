# -*- coding: utf-8 -*-


from collections import Counter
import re
from math import log,sqrt, ceil
import emoji
from tkinter import _flatten


class NER_Adaptive_Resampling():
    
    def __init__(self, inputpath, outputpath):
        self.inputpath = inputpath
        self.outputpath = outputpath
        
    
    def conll_data_read(self):
        
        # Load data in CoNLL format
        f = re.split('\n\t\n|\n\n|\n \n',open(self.inputpath,'r',encoding = 'utf-8').read())[:-1]
        x,y = [[] for i in range(len(f))],[[] for i in range(len(f))]
        for sen in range(len(f)):
            w = f[sen].split('\n')
            for line in w:
                        # Additional data cleaning: transform emoji into text, noisy text oridented.
                        x[sen].append(emoji.demojize(line.split(' ')[0]))
                        y[sen].append(line.split(' ')[-1])
        return x,y
    
    def get_stats(self):
        
        # Get stats of the class distribution of the dataset
        labels = list(_flatten(self.conll_data_read()[-1]))
        num_tokens = len(labels)
        ent = [label[2:] for label in labels if label != 'O']
        count_ent = Counter(ent)
        for key in count_ent:
            #Use frequency instead of count
            count_ent[key] = count_ent[key]/num_tokens
        return count_ent
    
    def resamp(self, method):
        
        # Select method by setting hyperparameters listed below:
        # sc: the smoothed resampling incorporating count
        # sCR: the smoothed resampling incorporating Count & Rareness
        # sCRD: the smoothed resampling incorporating Count, Rareness, and Density
        # nsCRD: the normalized and smoothed  resampling  incorporating Count, Rareness, and Density
        
        if method not in ['sc','sCR','sCRD','nsCRD']:
            raise ValueError("Unidentified Resampling Method")

        output = open(self.outputpath,'w',encoding = 'utf-8')
        x,y =  self.conll_data_read()
        stats = self.get_stats()
        
        
        
        for sen in range(len(x)):
            
            # Resampling time can at least be 1, which means sentence without 
            # entity will be reserved in the dataset  
            rsp_time = 1
            sen_len = len(y[sen])
            ents = Counter([label[2:] for label in y[sen] if label != 'O'])
                 # Pass if there's no entity in a sentence
            if ents:
                for ent in ents.keys():
                    # Resampling method selection and resampling time calculation, 
                    # see section 'Resampling Functions' in our paper for details.
                    if method == 'sc':
                        rsp_time += ents[ent]
                    if method == 'sCR' or method == 'sCRD':
                        weight = -log(stats[ent],2)
                        rsp_time += ents[ent]*weight
                    if method == 'nsCRD':
                        weight = -log(stats[ent],2)
                        rsp_time += sqrt(ents[ent])*weight
                if method == 'sCR':
                    rsp_time = sqrt(rsp_time)
                if method == 'sCRD' or method == 'nsCRD':
                    rsp_time = rsp_time/sqrt(sen_len)
                # Ceiling to ensure the integrity of resamling time
                rsp_time = ceil(rsp_time) 
            for t in range(rsp_time):
                for token in range(sen_len):
                    output.write(x[sen][token]+' '+y[sen][token]+'\n')
                output.write('\n')
        output.close()
                            
    def BUS(self):
        
        # Implementation of Balanced UnderSampling (BUS) mentioned in paper 
        # Balanced undersampling: a novel sentence-based undersampling method 
        # to improve recognition of named entities in chemical and biomedical text
        # Appl Intell (2018) Akkasi et al .
        
        # R parameter is set to 3, as what metioned in this paper.
        
        # Thank Jing Hou for pointing out a previous bug in this part
        
        
        output = open(self.outputpath,'w',encoding = 'utf-8')
        x,y =  self.conll_data_read()
        for sen in range(len(x)):
            num_sampled = len([label for label in y[sen] if label != 'O'])
            thres = 3*num_sampled
            mask = [1 if label != 'O' else 0 for label in y[sen] ]
            while num_sampled < thres and num_sampled < len(y[sen]):
                index=np.where(np.array(mask) == 1)[0]
                for i in index:
                    if i != len(mask)-1:
                        if mask[i+1] == 0:
                            mask[i+1] = 1
                            num_sampled += 1
                for i in index:
                    if i != 0:
                        if mask[i-1] == 0:
                            mask[i-1] = 1
                            num_sampled += 1
            for i in range(len(y[sen])):
                if mask[i] == 1:
                    output.write(x[sen][i]+' '+y[sen][i]+'\n')
            output.write('\n')
            
