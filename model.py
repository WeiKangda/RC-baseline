#!/usr/bin/env python
# coding: utf-8




# ## Load Dataset



from datasets import load_dataset, load_metric

#datasets = load_dataset('csv', data_files = {'train':[r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_1_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_2_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_3_context.csv',
#                                                      r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_4_context.csv'],
#                                             'test':r'C:\Users\hanso\Desktop\covid-faq\data\FAQ_Bank_eval_context.csv'})
train_ds = load_dataset('csv', data_files = r'/home/weikangda/xlm-baseline/data/test.csv', split='train[:70%]')
validation_ds = load_dataset('csv', data_files = r'/home/weikangda/xlm-baseline/data/test.csv', split='train[70%:85%]')
test_ds = load_dataset('csv', data_files = r'/home/weikangda/xlm-baseline/data/test.csv', split='train[85%:100%]')
#dataset = load_dataset('covid_qa_ucsd', 'zh', data_dir=r"C:\Users\hanso\Desktop\covid-faq\data")


# In[292]:


from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML
import json
import ast


def show_random_elements(dataset, num_examples=10):
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


# ## Data Preprocessing - COUGH Dataset

# In[293]:


from transformers import AutoTokenizer,BertTokenizer, BertModel
    
#tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
#tokenizer = BertTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"


# In[295]:


def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "index"],
        examples["index" if pad_on_right else "question"],
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
        answers = examples["answer_with_index"][sample_index]
        answers = ast.literal_eval(answers)
        
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


# In[296]:


tokenized_train_datasets = train_ds.map(prepare_train_features, batched=True, remove_columns=train_ds.column_names)
tokenized_validation_datasets = validation_ds.map(prepare_train_features, batched=True, remove_columns=validation_ds.column_names)
tokenized_test_datasets = test_ds.map(prepare_train_features, batched=True, remove_columns=test_ds.column_names)


# ## Model Fine-tuning

# In[297]:


from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

#model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
#model = BertModel.from_pretrained("deepset/xlm-roberta-large-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

# In[298]:


model


# In[308]:


args = TrainingArguments(
    f"test-cough",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    gradient_accumulation_steps=4
)


# In[300]:


from transformers import default_data_collator

data_collator = default_data_collator







# In[302]:


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "index"],
        examples["index" if pad_on_right else "question"],
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

    return tokenized_examples


# In[303]:


eval_features = validation_ds.map(
    prepare_validation_features,
    batched=True,
    remove_columns=validation_ds.column_names
)


# In[304]:


from tqdm.auto import tqdm
import collections
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["question"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["index"]
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
                        continue
                    # Don't consider answers with a length that is either < 0 
                    if end_index < start_index:
                        continue

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
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "","end":0,"start":0,"score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        predictions[example["question"]] = [best_answer["text"],best_answer["start"],best_answer["end"]]

    return predictions


# In[305]:


import statistics
f1s = []
def f1_eval(formatted_predictions, references):
    f1_scores = []
    for i,pair in enumerate(formatted_predictions):
        #print(i)
        predictions_words = formatted_predictions[i][0].split(' ')
        #print(type(formatted_predictions))
        references_words = references[i].split(' ')
        #print(references_words)
        p_number = 0
        r_number = 0
        for p_word in predictions_words:
            if p_word in references_words:
                p_number= p_number+1
        for r_word in references_words:
            if r_word in predictions_words:
                r_number=r_number+1
        precision = p_number / len(predictions_words)
        recall = r_number / len(references_words)
        if precision == 0 and recall == 0:
            f1_scores.append(0)
            continue
        #print(precision)
        #print(recall)
        f1 = 2*precision*recall/(precision+recall)
        f1_scores.append(f1)
        f1s.append(statistics.mean(f1_scores))
    #print(f1_scores)
    print('F1 score is: '+ str(statistics.mean(f1_scores)))


# In[309]:


import numpy as np
epoch= 10
for i in range(epoch):
    model = AutoModelForQuestionAnswering.from_pretrained("./test-cough-trained")
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_test_datasets,
        eval_dataset=tokenized_validation_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    raw_predictions = trainer.predict(eval_features)
    validation_predictions = postprocess_qa_predictions(validation_ds, eval_features, raw_predictions.predictions)
    formatted_predictions = [v for k, v in validation_predictions.items()]
    references = [ex["answer"] for ex in validation_ds]
    f1_eval(formatted_predictions, references)
    trainer.save_model("test-cough-trained")
print(f1s)
# In[ ]:


trainer.save_model("test-cough-trained")


# ## Evaluation

# In[133]:


import torch

for batch in trainer.get_eval_dataloader():
    break
batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
    output = trainer.model(**batch)
output.keys()


# In[134]:


output.start_logits.shape, output.end_logits.shape


# In[135]:


output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)


# In[ ]:


validation_features = test_ds.map(
    prepare_validation_features,
    batched=True,
    remove_columns=test_ds.column_names
)


# In[139]:


raw_predictions = trainer.predict(validation_features)


# In[140]:


validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))


# In[276]:


test_ds[0]["answer"]


# In[142]:


import collections

examples = test_ds
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["question"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)


# In[229]:


final_predictions = postprocess_qa_predictions(test_ds, validation_features, raw_predictions.predictions)


# In[230]:


print(final_predictions)


# In[231]:


metric = load_metric("f1")


# In[232]:


print(metric.description)


# In[233]:


formatted_predictions = [v for k, v in final_predictions.items()]
print(formatted_predictions)
references = [ex["answer"] for ex in test_ds]
print(references)


# In[237]:


f1_eval(formatted_predictions, references)

