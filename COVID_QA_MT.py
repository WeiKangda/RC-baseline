#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset, load_metric,concatenate_datasets

#datasets = load_dataset('csv', data_files = {'train':[r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_1_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_2_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_3_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_4_context.csv'],
#                                             'test':r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_eval_context.csv'})
#train_ds = load_dataset('csv', data_files = r'C:\Users\hanso\Desktop\covid-faq\data\test.csv', split='train[:70%]')
#validation_ds = load_dataset('csv', data_files = r'C:\Users\hanso\Desktop\covid-faq\data\test.csv', split='train[70%:85%]')
#test_ds = load_dataset('csv', data_files = r'C:\Users\hanso\Desktop\covid-faq\data\test.csv', split='train[85%:100%]')
#dataset = load_dataset('covid_qa_deepset')
dataset = load_dataset('covid_qa_deepset', split='train[:100%]')
test_ds = load_dataset('covid_qa_deepset', split='train[:1]')



from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML
import json
import ast


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
print(test_ds)
show_random_elements(test_ds)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()         

cuda = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
#tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
#model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model.to(cuda)

from tqdm.auto import tqdm
import numpy as np
import csv
def translate(data):
    translated_data = {'answers':[], 'context':[], 'document_id':[], 'id':[], 'is_impossible':[], 'question':[]} 
    document_id = -1
    story_p = ''
    for i,row in enumerate(tqdm(data)):
        translated_data['document_id'].append(data['document_id'][i])
        translated_data['id'].append(data['id'][i])
        translated_data['is_impossible'].append(data['is_impossible'][i])
        
        
        tokenized_qa = tokenizer.prepare_seq2seq_batch([data[i]["answers"]["text"][0],data[i]["question"]], return_tensors='pt')
        translation_qa = model.generate(**tokenized_qa.to(cuda))
        translated_qa = tokenizer.batch_decode(translation_qa, skip_special_tokens=False)
        print(translated_qa)
        translated_question = translated_qa[1].replace('<pad>','')
        translated_answer = translated_qa[0].replace('<pad>','')
        answer = {'answer_start':data['answers'][i]['answer_start'], 'text':[translated_answer]}
        translated_data['answers'].append(answer)
        translated_data['question'].append(translated_question)
        
        if document_id == data[i]['document_id']:
            translated_data['context'].append(story_p)
            document_id = data[i]['document_id']
        else:
            context = data[i]["context"].split(".")
            print(context)
            
            #print(translated_context)
        
            story = ''
            for sentence in context:
                tokenized_sentence = tokenizer.prepare_seq2seq_batch([sentence], return_tensors='pt')
                translation_sentence = model.generate(**tokenized_sentence.to(cuda))
                translated_sentence = tokenizer.batch_decode(translation_sentence, skip_special_tokens=False)
                #print(len(translated_sentence))
                translated_sentence = translated_sentence[0].replace('<pad>','')
                story = story + translated_sentence + '。'
            translated_data['context'].append(story)
            story_p = story
            document_id = data[i]['document_id']
        
        
        df = pd.DataFrame(translated_data,columns=['answers', 'context', 'document_id', 'id', 'is_impossible', 'question'])
        df.to_csv(r'COVID_QA_es.csv', index=False)
    
    
translate(dataset)

