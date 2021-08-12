#!/usr/bin/env python
# coding: utf-8


# ## Load Dataset



from datasets import load_dataset, load_metric, concatenate_datasets

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
train_ar_ds = load_dataset('xquad', 'xquad.ar',split='validation[0%:70%]')
train_de_ds = load_dataset('xquad', 'xquad.de',split='validation[0%:70%]')
train_zh_ds = load_dataset('xquad', 'xquad.zh',split='validation[0%:70%]')
train_vi_ds = load_dataset('xquad', 'xquad.vi',split='validation[0%:70%]')
train_en_ds = load_dataset('xquad', 'xquad.en',split='validation[0%:70%]')
train_es_ds = load_dataset('xquad', 'xquad.es',split='validation[0%:70%]')
train_hi_ds = load_dataset('xquad', 'xquad.hi',split='validation[0%:70%]')
train_el_ds = load_dataset('xquad', 'xquad.el',split='validation[0%:70%]')
train_th_ds = load_dataset('xquad', 'xquad.th',split='validation[0%:70%]')
train_tr_ds = load_dataset('xquad', 'xquad.tr',split='validation[0%:70%]')
train_ru_ds = load_dataset('xquad', 'xquad.ru',split='validation[0%:70%]')
train_ro_ds = load_dataset('xquad', 'xquad.ro',split='validation[0%:70%]')

va_ar_ds = load_dataset('xquad', 'xquad.ar',split='validation[70%:85%]')
va_de_ds = load_dataset('xquad', 'xquad.de',split='validation[70%:85%]')
va_zh_ds = load_dataset('xquad', 'xquad.zh',split='validation[70%:85%]')
va_vi_ds = load_dataset('xquad', 'xquad.vi',split='validation[70%:85%]')
va_en_ds = load_dataset('xquad', 'xquad.en',split='validation[70%:85%]')
va_es_ds = load_dataset('xquad', 'xquad.es',split='validation[70%:85%]')
va_hi_ds = load_dataset('xquad', 'xquad.hi',split='validation[70%:85%]')
va_el_ds = load_dataset('xquad', 'xquad.el',split='validation[70%:85%]')
va_th_ds = load_dataset('xquad', 'xquad.th',split='validation[70%:85%]')
va_tr_ds = load_dataset('xquad', 'xquad.tr',split='validation[70%:85%]')
va_ru_ds = load_dataset('xquad', 'xquad.ru',split='validation[70%:85%]')
va_ro_ds = load_dataset('xquad', 'xquad.ro',split='validation[70%:85%]')

te_ar_ds = load_dataset('xquad', 'xquad.ar',split='validation[85%:100%]')
te_de_ds = load_dataset('xquad', 'xquad.de',split='validation[85%:100%]')
te_zh_ds = load_dataset('xquad', 'xquad.zh',split='validation[85%:100%]')
te_vi_ds = load_dataset('xquad', 'xquad.vi',split='validation[85%:100%]')
te_en_ds = load_dataset('xquad', 'xquad.en',split='validation[85%:100%]')
te_es_ds = load_dataset('xquad', 'xquad.es',split='validation[85%:100%]')
te_hi_ds = load_dataset('xquad', 'xquad.hi',split='validation[85%:100%]')
te_el_ds = load_dataset('xquad', 'xquad.el',split='validation[85%:100%]')
te_th_ds = load_dataset('xquad', 'xquad.th',split='validation[85%:100%]')
te_tr_ds = load_dataset('xquad', 'xquad.tr',split='validation[85%:100%]')
te_ru_ds = load_dataset('xquad', 'xquad.ru',split='validation[85%:100%]')
te_ro_ds = load_dataset('xquad', 'xquad.ro',split='validation[85%:100%]')


train_ds = concatenate_datasets([train_ar_ds,train_de_ds,train_zh_ds,train_vi_ds,train_en_ds,train_es_ds,train_hi_ds,train_el_ds,train_th_ds,train_tr_ds,train_ru_ds,train_ro_ds])
validation_ds = concatenate_datasets([va_ar_ds,va_de_ds,va_zh_ds,va_vi_ds,va_en_ds,va_es_ds,va_hi_ds,va_el_ds,va_th_ds,va_tr_ds,va_ru_ds,va_ro_ds])
test_ds = concatenate_datasets([te_ar_ds,te_de_ds,te_zh_ds,te_vi_ds,te_en_ds,te_es_ds,te_hi_ds,te_el_ds,te_th_ds,te_tr_ds,te_ru_ds,te_ro_ds])
#train_ds = load_dataset('covid_qa_deepset', split='train[:70%]')
#validation_ds = load_dataset('covid_qa_deepset', split='train[70%:85%]')
#test_ds = load_dataset('covid_qa_deepset', split='train[0%:100%]')



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
    
print(train_ds)
print(validation_ds)
print(test_ds)
show_random_elements(train_ds)


# ## Data Preprocessing

# In[46]:


from transformers import AutoTokenizer,BertTokenizer, XLMTokenizer, XLMModel
    
#tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
#tokenizer = BertTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
#tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')


# In[9]:


max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"


# In[51]:


def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    #print(examples)
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    #print(sample_mapping)
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    #print(offset_mapping)
    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        #print(cls_index)
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        #answers = ast.literal_eval(answers)
        #print(answers)
        # If no answers are given, set the cls_index as answer.
        if len(answers) == 0:
            #print('no answers')
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            #start_char = answers[0]
            #rint(start_char)
            #end_char = start_char + len(answers[1])
            start_char = answers["answer_start"][0]
            print(start_char)
            end_char = start_char + len(answers["text"][0])
            print(end_char)
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                #print('answer out of span')
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    print(tokenized_examples)
    return tokenized_examples

def prepare_test_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    #print(examples)
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        #answers = ast.literal_eval(answers)
        
        # If no answers are given, set the cls_index as answer.
        if len(answers) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            #start_char = answers[0]
            #rint(start_char)
            #end_char = start_char + len(answers[1])
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    print(tokenized_examples)
    return tokenized_examples
# In[52]:


tokenized_train_datasets = train_ds.map(prepare_train_features, batched=True, remove_columns=train_ds.column_names)
tokenized_validation_datasets = validation_ds.map(prepare_train_features, batched=True, remove_columns=validation_ds.column_names)
tokenized_test_datasets = test_ds.map(prepare_test_features, batched=True, remove_columns=test_ds.column_names)
#tokenzied_sample_datasets = test_ds.map(prepare_train_features, batched=True, remove_columns=sample_ds.column_names)


# ## Model Fine-tuning

# In[27]:


from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("./test-xquad-trained-xlm-best")
#model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
#model = BertModel.from_pretrained("mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp")
#model = ModelForQuestionAnswering.from_pretrained("mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp")


# In[28]:


model


# In[29]:


args = TrainingArguments(
    f"test-xquad",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    gradient_accumulation_steps=4
)


# In[30]:


from transformers import default_data_collator

data_collator = default_data_collator


# In[31]


# In[33]:


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["question"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    print(tokenized_examples)
    return tokenized_examples




eval_features = validation_ds.map(
    prepare_validation_features,
    batched=True,
    remove_columns=validation_ds.column_names
)


# In[72]:


from tqdm.auto import tqdm
import collections
import random
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["question"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()
    #predictions = {}
    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    count = 0
    for example_index, example in enumerate(tqdm(examples)):
        count = count + 1
        print(example_index)
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        valid_answers.append(
                            {
                                "score": 0,
                                "start": 0,
                                "end": 0,
                                "text": ""
                            }
                        )
                    # Don't consider answers with a length that is either < 0 
                    elif end_index < start_index:
                        valid_answers.append(
                            {
                                "score": 0,
                                "start": 0,
                                "end": 0,
                                "text": ""
                            }
                        )

                    else:
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start": start_char,
                                "end": end_char,
                                "text": context[start_char: end_char]
                            }
                        )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            #print(best_answer)
            if example["question"] in predictions:
                predictions[example["question"] +str(random.randint(0,99999))] = [best_answer["text"],best_answer["start"],best_answer["end"]]
            else:
                predictions[example["question"]] = [best_answer["text"],best_answer["start"],best_answer["end"]]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            
            best_answer = {'score': 0, 'start': 0, 'end': 0, 'text': ''}
            if example["question"] in predictions:
                predictions[example["question"] +str(random.randint(0,99999))] = [best_answer["text"],best_answer["start"],best_answer["end"]]
            else:
                predictions[example["question"]] = [best_answer["text"],best_answer["start"],best_answer["end"]]
        
    return predictions


# In[73]:

import collections
import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Normalize string and split string into tokens."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute the Exact Match score."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1_from_tokens(gold_toks: List[str], pred_toks: List[str]) -> float:
    """Compute the F1 score from tokenized gold answer and prediction."""
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute the F1 score."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    return compute_f1_from_tokens(gold_toks, pred_toks)

import statistics
def f1_eval(formatted_predictions, references):
    f1_scores = []
    for i,pair in enumerate(formatted_predictions):
        #print(i)
        f1 = compute_f1(formatted_predictions[i],references[i])
        f1_scores.append(f1)
    #print(f1_scores)
    print('F1 score is: '+str(statistics.mean(f1_scores)))
    return statistics.mean(f1_scores)

# In[93]




import numpy as np
epoch= 0
f1 = 0
best_model_id = 0
for i in range(epoch):
    if i != 0:
        model = AutoModelForQuestionAnswering.from_pretrained("./test-xquad-trained-xlm")
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_validation_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    raw_predictions = trainer.predict(eval_features)
    validation_predictions = postprocess_qa_predictions(validation_ds, eval_features, raw_predictions.predictions)
    formatted_predictions = [v[0] for k, v in validation_predictions.items()]
    references = [ex["answers"]["text"][0] for ex in validation_ds]
    new_f1 = f1_eval(formatted_predictions, references)
    trainer.save_model("./test-xquad-trained-xlm")
    if new_f1 > f1:
        f1 = new_f1
        trainer.save_model("./test-xquad-trained-xlm-best")
        best_model_id = i
        print("The best trained model so far is trained at epoch NO."+str(best_model_id))

# In[ ]:
model = AutoModelForQuestionAnswering.from_pretrained("./test-xquad-trained-xlm-best")
trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_validation_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
)
print("The best trained model is trained at epoch NO."+str(best_model_id))

# ## Evaluation

# In[23]:


import torch

for batch in trainer.get_eval_dataloader():
    break
batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
    output = trainer.model(**batch)
output.keys()


# In[24]:


output.start_logits.shape, output.end_logits.shape


# In[25]:


output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)


# In[26]:


validation_features = test_ds.map(
    prepare_validation_features,
    batched=True,
    remove_columns=test_ds.column_names
)


# In[27]:


raw_predictions = trainer.predict(validation_features)


# In[28]:


validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))


# In[11]:



# In[31]:






# In[2]:


import numpy as np
final_predictions = postprocess_qa_predictions(test_ds, validation_features, raw_predictions.predictions)


# In[35]:


print(final_predictions)


formatted_predictions = [v[0] for k, v in final_predictions.items()]
print(formatted_predictions)
references = [ex["answers"]["text"][0] for ex in test_ds]
@print(references)
f1_eval(formatted_predictions, references)


# In[237]:
#formatted_predictions = [v for k, v in final_predictions.items()]
#print(formatted_predictions)
#references = [ex["answer"] for ex in test_ds]
#print(references)

output = {'prediction': formatted_predictions}
df = pd.DataFrame(output)
df.to_csv(r'output.csv', index=False)
