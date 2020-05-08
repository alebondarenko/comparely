from pytorch_transformers import BertTokenizer, BertForTokenClassification
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
from seq_lab.LSTMCRF_seq_lab.CRF import CRF
from seq_lab.LSTMCRF_seq_lab.LSTMCRFTagger import LSTMCRFTagger
import pickle
from nltk.tokenize import TreebankWordTokenizer
from tqdm.notebook import tqdm


class SeqLabeller:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name == 'lstmcrftagger':
            self.tag2ix = {
                '<pad>': 0,
                'O': 1,
                'B-OBJ': 2,
                'I-OBJ': 3,
                'B-PREDFULL': 4,
                'I-PREDFULL': 5
            }
            self.model = LSTMCRFTagger(torch.zeros(400002, 300), 128, len(self.tag2ix), self.device)
            self.model.load_state_dict(torch.load('seq_lab/LSTMCRF_seq_lab/pytorch_model.bin'))
            self.model.to(self.device)
#             self.model = torch.load('seq_lab/LSTMCRF_seq_lab/bilstm+crf.bin')
            with open('seq_lab/LSTMCRF_seq_lab/vocab.pkl', 'rb') as f:
                self.vocab = pickle.load(f)
        elif model_name == 'berttagger':
            self.tag2ix = {
                '<pad>': 0,
                'O': 1,
                'B-OBJ': 2,
                'B-PREDFULL': 3,
                'I-PREDFULL': 4,
                'I-OBJ': 5
            }
            
#             self.CACHE_DIR = 'cache/'
            self.MODEL_DIR = "seq_lab/BERT_seq_lab/"
            self.bpe_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
#             self.BPE_TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', cache_dir=CACHE_DIR, do_lower_case=False)
            self.MAX_LEN = 80
            self.model = BertForTokenClassification.from_pretrained(self.MODEL_DIR, num_labels=len(self.tag2ix)).to(self.device)
            
        self.ix2tag = {value: key for key, value in self.tag2ix.items()}
        self.PRED_BATCH_SIZE = 1
            
    def bpe_tokenize(self, words):
        new_words = []
        bpe_masks = []
        for word in words:
            bpe_tokens = self.bpe_tokenizer.tokenize(word)
            new_words += bpe_tokens        
            bpe_masks += [1] + [0] * (len(bpe_tokens) - 1)

        return new_words, bpe_masks

    def add_labels_for_bpe_suffix(self, labels, bpe_masks):
        result_labels = []
        for l_sent, m_sent in zip(labels, bpe_masks):
            m_sent = m_sent
            sent_res = []
            i = 0
            for l in l_sent:
                sent_res.append(l)
    
                i += 1
                while i < len(m_sent) and (m_sent[i] == 0):
                    i += 1
                    sent_res.append('<pad>')

            result_labels.append(sent_res)

        return result_labels

    def prepare_bpe_tokens_for_bert(self, tokens):
        return [['[CLS]'] + list(toks[:self.MAX_LEN - 2]) + ['[SEP]'] for toks in tokens]


    def prepare_bpe_labels_for_bert(self, labels):
        return [['<pad>'] + list(ls[:self.MAX_LEN - 2]) + ['<pad>'] for ls in labels]

    def generate_masks(self, input_ids):
        return input_ids > 0

    def make_tokens_tensors(self, tokens):
        bpe_tokens, bpe_masks = tuple(zip(*[self.bpe_tokenize(sent) for sent in tokens]))
        bpe_tokens = self.prepare_bpe_tokens_for_bert(bpe_tokens)
        bpe_masks = [masks[:self.MAX_LEN-2] for masks in bpe_masks]
        max_len = max(len(sent) for sent in bpe_tokens)
        token_ids = torch.tensor(pad_sequences([self.bpe_tokenizer.convert_tokens_to_ids(sent) for sent in bpe_tokens], 
                                               maxlen=max_len, dtype='long', 
                                               truncating='post', padding='post'))
        token_masks = self.generate_masks(token_ids)
        return bpe_tokens, max_len, token_ids, token_masks, bpe_masks
    
    
    def make_label_tensors(self, labels, bpe_masks, max_len):
        bpe_labels = self.add_labels_for_bpe_suffix(labels, bpe_masks)
        bpe_labels = self.prepare_bpe_labels_for_bert(bpe_labels)
    
        label_ids = pad_sequences([[self.tag2ix.get(l) for l in lab] for lab in bpe_labels],
                             maxlen=max_len, value=self.tag2ix['<pad>'], padding='post',
                             dtype='long', truncating='post')
        return torch.LongTensor(label_ids)

    def generate_tensors_for_training(self, tokens, labels):
        _, max_len, token_ids, token_masks, bpe_masks = self.make_tokens_tensors(tokens)
        label_ids = self.make_label_tensors(labels, bpe_masks, max_len)
        return token_ids, token_masks, label_ids

    def generate_tensors_for_prediction(self, evaluate, dataset_row):
        dataset_row = dataset_row
        labels = None
        if evaluate:
            tokens, labels = tuple(zip(*dataset_row))
        else:
            tokens = dataset_row

        _, max_len, token_ids, token_masks, bpe_masks = self.make_tokens_tensors(tokens)
        label_ids = None

        if evaluate:
            label_ids = self.make_label_tensors(labels, bpe_masks, max_len)

        return token_ids, token_masks, bpe_masks, label_ids, tokens, labels
    
    def logits_to_preds(self, logits, bpe_masks, tokens):
        preds = logits.argmax(dim=2).numpy()
        probs = logits.numpy().max(axis=2)
        prob = [np.mean([p for p, m in zip(prob[1:-1][:len(masks)], masks[:len(prob)-2]) if m])  
                for prob, masks in zip(probs, bpe_masks)]
        preds = [[self.ix2tag[p] for p, m in zip(pred[1:-1][:len(masks)], masks[:len(pred)-2]) if m] 
                 for pred, masks in zip(preds, bpe_masks)]
        preds = [pred + ['O']*(max(0, len(toks) - len(pred))) for pred, toks in zip(preds, tokens)]
        return preds, prob
    
    def bert_predict(self, dataset, evaluate=False, metrics=None, pred_loader_args={'num_workers' : 1}):
        if metrics is None:
            metrics = []

        self.model.eval()

        dataloader = DataLoader(dataset, 
                                collate_fn=lambda dataset_row: self.generate_tensors_for_prediction(evaluate, dataset_row), 
                                **pred_loader_args, 
                                batch_size=self.PRED_BATCH_SIZE)

        predictions = []
        probas = []

        if evaluate:
            cum_loss = 0.
            true_labels = []

        for nb, tensors in enumerate(dataloader):
            token_ids, token_masks, bpe_masks, label_ids, tokens, labels = tensors

            if evaluate:
                true_labels.extend(labels)

            with torch.no_grad():
                token_ids = token_ids.cuda()
                token_masks = token_masks.cuda()

                if evaluate:
                    label_ids = label_ids.cuda()

                logits = self.model(token_ids, 
                                    token_type_ids=None,
                                    attention_mask=token_masks,
                                    labels=label_ids,)

                if evaluate:
                    loss, logits = logits
                    cum_loss += loss.mean().item()
                else:
                    logits = logits[0]

                b_preds, b_prob = self.logits_to_preds(logits.cpu(), bpe_masks, tokens)

            predictions.extend(b_preds)
            probas.extend(b_prob)

        if evaluate: 
            cum_loss /= (nb + 1)

            result_metrics = []
            for metric in metrics:
                result_metrics.append(metric(true_labels, predictions))

            return predictions, probas, tuple([cum_loss] + result_metrics)
        else:
            return predictions, probas

    def make_tensors_predict(self, batch):
        tokens = batch
        pad = self.vocab['<pad>']
        lines_ix = []
        seq_lens = []
        for i in range(len(tokens)):
            line_ix = [self.vocab.get(l, self.vocab['UNK']) for l in tokens[i]]
            lines_ix.append(torch.LongTensor(line_ix))
            seq_lens.append(len(line_ix))
        tensor_x = pad_sequence(lines_ix, batch_first=True, padding_value=pad)
        return tensor_x, seq_lens

    def lstmcrf_predict(self, input_dataset):
        preds = []
        self.model.eval()
        data_loader = DataLoader(input_dataset, batch_size=self.PRED_BATCH_SIZE, collate_fn=self.make_tensors_predict)
        for sample in tqdm(data_loader):
            features, seq_lens = sample
            features = features.to(self.device)
            with torch.no_grad():
                features_rnn = self.model.forward_birnn(features, seq_lens, self.device)
            
                mask = self.model.get_mask_from_lens(seq_lens, self.device)
                y_pred = self.model.crf.decode_viterbi(features_rnn, mask, self.device)
#                 y_pred = y_pred.tolist()
                for i, sent_len in enumerate(seq_lens):
                    y_pred[i] = y_pred[i][:sent_len]
                y_pred = [[self.ix2tag[w] for w in sent] for sent in y_pred]
                preds += y_pred
    
        return preds
    
    def get_labels(self, sentences):
        tokens = list(map(TreebankWordTokenizer().tokenize, sentences))
        if self.model_name == 'lstmcrftagger':
            return tokens, self.lstmcrf_predict(tokens)
        elif self.model_name == 'berttagger':
            preds, _ = self.bert_predict(tokens)
            return tokens, preds