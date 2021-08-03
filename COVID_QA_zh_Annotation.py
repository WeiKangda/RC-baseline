#!/usr/bin/env python
# coding: utf-8


from datasets import load_dataset, load_metric
dataset = load_dataset('csv', data_files = r'COVID_QA_zh.csv',split='train[:100%]')
# dataset = load_dataset('csv', data_files = r'COVID_QA_zh.csv',split='train[:2%]')
#example = load_dataset('csv', data_files = r'C:\Users\hanso\Desktop\Research_Folder\COVID-19_QA_Phase_B\COVID_QA_zh.csv',split='train[1:2]')


import time
import argparse
from datasets import ClassLabel, Sequence
import random
import pandas as pd
import json
import ast


import numpy as np
def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])

import ast
#print(type(example[0]['answers']))
#print(ast.literal_eval(example[0]["answers"]))
#print(ast.literal_eval(example[0]["answers"])["text"][0])
#print(example[0]['context'][0:10])

from tqdm.auto import tqdm
import numpy as np
import csv

def find_answer_index(data,num_splits,split_num):
    save_path=f"COVID_QA_zh_annotated_split_{split_num}_of_{num_splits}"
    ds = {'answers':[], 'context':[], 'document_id':[], 'id':[], 'is_impossible':[], 'question':[]}
    if split_num==0:
        iterator=tqdm(data)
    else:
        iterator=data
    for i,row in enumerate(iterator):
        ds['document_id'].append(data['document_id'][i])
        ds['id'].append(data['id'][i])
        ds['is_impossible'].append(data['is_impossible'][i])
        ds['context'].append(data['context'][i])
        ds['question'].append(data['question'][i])
        
        answer_index = -1
        ratio_p = 0
        answer = ast.literal_eval(data["answers"][i])["text"][0]
        answer_length = len(answer)
        skip_next=False
        skip_length=3
        for j in range(0,len(data['context'][i]),skip_length):
            if skip_next:
                skip_next=False
                continue
            ratio = levenshtein_ratio_and_distance(answer, data['context'][i][j:(j+ answer_length)], ratio_calc = True)
            if ratio > ratio_p:
                answer_index = j
                ratio_p = ratio
            
            if ratio >= ratio_p*0.9:
                for k in range(1,skip_length):
                    ratio = levenshtein_ratio_and_distance(answer, data['context'][i][j+k:(j+ answer_length+k)], ratio_calc = True)
                    if ratio > ratio_p:
                        answer_index=j+k
                        ratio_p=ratio
            else:
                skip_next=True

        # original method
        # ratio_p=0
        # answer_index = -1
        # start = time.time()
        # for j in range(0,len(data['context'][i]),2):
        #     ratio = levenshtein_ratio_and_distance(answer, data['context'][i][j:(j+ answer_length)], ratio_calc = True)
        #     # print(ratio)
        #     if ratio > ratio_p:
        #         answer_index = j
        #         ratio_p = ratio


        answer_annotated = {'answer_start':[], 'text':[]}
        answer_annotated['answer_start'].append(answer_index)
        answer_annotated['text'].append(ast.literal_eval(data['answers'][i])['text'][0])
        ds['answers'].append(answer_annotated)
        df = pd.DataFrame(ds,columns=['answers', 'context', 'document_id', 'id', 'is_impossible', 'question'])
        df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_splits',type=int, default=8)
    parser.add_argument('--split_num',type=int,required=True)
    args = parser.parse_args()
    split_size = round(100/args.num_splits)
    start_percent = split_size * args.split_num
    end_percent = min(100,start_percent + split_size)

    # print(f"start: {start_percent} - end: {end_percent}")
    dataset = dataset = load_dataset('csv', data_files = r'COVID_QA_zh.csv',split=f'train[{start_percent}%:{end_percent}%]')
    find_answer_index(dataset,args.num_splits,args.split_num)