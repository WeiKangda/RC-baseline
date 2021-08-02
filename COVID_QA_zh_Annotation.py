#!/usr/bin/env python
# coding: utf-8

# In[42]:


from datasets import load_dataset, load_metric

dataset = load_dataset('csv', data_files = r'/home/weikangda/xlm-baseline/COVID_QA_zh_1.csv',split='train[:100%]')
#example = load_dataset('csv', data_files = r'C:\Users\hanso\Desktop\Research_Folder\COVID-19_QA_Phase_B\COVID_QA_zh.csv',split='train[1:2]')





from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML
import json
import ast
from numba import jit, cuda

def show_random_elements(dataset, num_examples=1):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
    
print(dataset)
#print(example)
#show_random_elements(example)


# In[38]:


import numpy as np
#@jit(target ="cuda")
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


# In[35]:


import ast
#print(type(example[0]['answers']))
#print(ast.literal_eval(example[0]["answers"]))
#print(ast.literal_eval(example[0]["answers"])["text"][0])
#print(example[0]['context'][0:10])


# In[44]:


from tqdm.auto import tqdm
import numpy as np
import csv

#@jit(target ="cuda")
def find_answer_index(data):
    ds = {'answers':[], 'context':[], 'document_id':[], 'id':[], 'is_impossible':[], 'question':[]} 
    
    for i,row in enumerate(tqdm(data)):
        ds['document_id'].append(data['document_id'][i])
        ds['id'].append(data['id'][i])
        ds['is_impossible'].append(data['is_impossible'][i])
        ds['context'].append(data['context'][i])
        ds['question'].append(data['question'][i])
        
        answer_index = -1
        ratio_p = 0
        answer = ast.literal_eval(data["answers"][i])["text"][0]
        answer_length = len(answer)
        for j,character in enumerate(data['context'][i]):
            ratio = levenshtein_ratio_and_distance(answer, data['context'][i][j:(j+ answer_length)], ratio_calc = True)
            if ratio > ratio_p:
                answer_index = j
                ratio_p = ratio
        answer_annotated = {'answer_start':[], 'text':[]}
        answer_annotated['answer_start'].append(answer_index)
        answer_annotated['text'].append(ast.literal_eval(data['answers'][i])['text'][0])
        ds['answers'].append(answer_annotated)
        df = pd.DataFrame(ds,columns=['answers', 'context', 'document_id', 'id', 'is_impossible', 'question'])
        df.to_csv(r'COVID_QA_zh_annotated.csv', index=False)
        #print(df)
    
find_answer_index(dataset)

