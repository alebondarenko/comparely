import torch
import joblib
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer
from multiprocessing import Pool, cpu_count
from tqdm.notebook import tqdm as tqdm_notebook

class CompSentClf:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'bow+xgboost':
            self.model = joblib.load('bow/bow+xgboost.bin')
            self.classes = self.model.classes_
        elif model_name == 'infersent+xgboost':
            self.model = joblib.load('infersent/infersent+xgboost.bin')
            self.classes = self.model.classes_
        elif model_name == 'elmo+linreg':
            self.model = joblib.load('elmo/elmo+linreg.bin')
            self.classes = self.model.classes_
        elif model_name == 'bert':
            self.classes = ["BETTER", "WORSE", "NONE"]
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert/vocab.txt', do_lower_case=False)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.MAX_SEQ_LENGTH = 78
            self.EVAL_BATCH_SIZE = 32
    
    def get_proba(self, sentences):
        if self.model_name == 'bert':
            examples_for_processing = [(example[1], self.MAX_SEQ_LENGTH, self.bert_tokenizer)
                                       for example in sentences.iterrows()]
            process_count = cpu_count() - 1
            print(f'Preparing to convert {len(sentences)} examples..')
            print(f'Spawning {process_count} processes..')
            with Pool(process_count) as p:
                features = list(tqdm_notebook(p.imap(self.convert_example_to_bert_feature,
                                                     examples_for_processing),
                                              total=len(sentences)))
            all_input_ids, all_input_mask, all_segment_ids = self.get_bert_tensors(features)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.EVAL_BATCH_SIZE)
            
            model = BertForSequenceClassification.from_pretrained('bert/', cache_dir='cache/',
                                                                       num_labels=len(self.classes))
            model.to(self.device)
            
            probas = None
            
            for input_ids, input_mask, segment_ids in tqdm_notebook(eval_dataloader, desc="Predicting"):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)
                    prob = torch.nn.functional.softmax(logits, dim=1)
                
                if probas is None:
                    probas = prob.detach().cpu().numpy()
                else:
                    probas = np.append(probas, prob.detach().cpu().numpy(), axis=0)
            return probas
        else:
            return self.model.predict_proba(sentences)
    
    def classify_sentences(self, sentences):
        df = pd.DataFrame(self.get_proba(sentences), columns=self.classes)
        df['max'] = df.idxmax(axis=1)
        return df
    
    def convert_example_to_bert_feature(self, example_row):
        example, max_seq_length, tokenizer = example_row
        tokens_a = tokenizer.tokenize(example["sentence"])
        tokens_b = tokenizer.tokenize(example["object_a"] + " " + example["object_b"])
        
        total_length = len(tokens_a) + len(tokens_b) + 3
        if total_length > max_seq_length:
            tokens_a = tokens_a[:(max_seq_length - (len(tokens_b) + 3))]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids}
    
    def get_bert_tensors(self, features, with_labels=False):
        all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f["input_mask"] for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f["segment_ids"] for f in features], dtype=torch.long)
        if with_labels:
            all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)
            return all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        else:
            return all_input_ids, all_input_mask, all_segment_ids