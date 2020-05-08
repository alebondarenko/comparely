import requests
from requests.auth import HTTPBasicAuth
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
import joblib

from CompSentClf import CompSentClf
from pke.unsupervised import MultipartiteRank
from w2v.w2v_feature import initialize_w2v, W2VFeature
from seq_lab.SeqLabeller import SeqLabeller

class Pipeline:
    def __init__(self, comp_sent_clf_name, seq_labeller_name=None):
        self.name = "reader"
        self.password = "reader"
        self.comp_sent_clf_name = comp_sent_clf_name
        self.seq_labeller_name = seq_labeller_name
        
        print("Initializing comparative sentences classifier")
        self.comp_sent_clf = CompSentClf(comp_sent_clf_name)
        if seq_labeller_name is None:
            print("Initializing aspect classifier")
            w2v_model = initialize_w2v()
            self.asp_clf = make_pipeline(W2VFeature(w2v_model), joblib.load('w2v/asp_clf.pkl'))
        else:
            print("Initializing sequence labeller")
            self.seq_labeller = SeqLabeller(seq_labeller_name)
            
    def get_structured_answer(self, obj_a, obj_b):
        obj_a = obj_a.lower().strip()
        obj_b = obj_b.lower().strip()
        
        print("Requesting Elasticsearch")
        json_compl = request_elasticsearch(obj_a, obj_b, self.name, self.password)
        
        print("Preparing sentences")
        all_sentences = extract_sentences(json_compl)
        remove_questions(all_sentences)
        prepared_sentences = prepare_sentence_DF(all_sentences, obj_a, obj_b)
        
        print("Classifying comparative sentences")
        classification_results = self.comp_sent_clf.classify_sentences(prepared_sentences)
        comparative_sentences = prepared_sentences[classification_results['max'] != 'NONE']
        comparative_sentences['max'] = classification_results[classification_results['max'] != 'NONE']['max']
        
        if self.seq_labeller_name is None:
            print("Looking for keyphrases")
            text = prepared_sentences[classification_results['max'] != 'NONE']['sentence'].str.cat(sep=' ')
            extractor = MultipartiteRank()
            extractor.load_document(input=text, language="en", normalization='stemming')
            extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=-1, stemming=False)
            asp_df = pd.DataFrame(columns=['object_a', 'object_b', 'aspect', 'sentence', 'max'])
            forbidden_phrases = [obj_a, obj_b, 'better', 'worse']
            
            for index, row in comparative_sentences.iterrows():
                sentence = row['sentence']
                for (keyphrase, score) in keyphrases:
                    skip_keyphrase = False
                    for phrase in forbidden_phrases:
                        if keyphrase == phrase:
                            skip_keyphrase = True
                            break
                    if not skip_keyphrase:
                        if keyphrase in sentence:
                            asp_df = asp_df.append(
                                {'object_a': row['object_a'],
                                 'object_b': row['object_b'],
                                 'aspect': keyphrase,
                                 'sentence': row['sentence'],
                                 'max': row['max'],
                                }, ignore_index=True)
            print("Predicting good aspects")
            y_pred = self.asp_clf.predict(asp_df)
            aspects = asp_df.iloc[np.nonzero(y_pred)[0].tolist()]['aspect'].unique()
        else:
            print("Sequence labelling")
            sentences = prepared_sentences[classification_results['max'] != 'NONE']['sentence'].tolist()
            words, preds = self.seq_labeller.get_labels(sentences)
            asp_df = pd.DataFrame(columns=['object_a', 'object_b', 'aspect', 'sentence', 'max'])
            aspects = set()
            for i, sent in enumerate(words):
                for j, word in enumerate(sent):
                    if preds[i][j] == 'B-PREDFULL':
                        cur_asp = word
                        for k in range(j + 1, len(sent)):
                            if preds[i][k] == 'I-PREDFULL':
                                cur_asp = cur_asp + ' ' + sent[k]
                            else:
                                break
                        aspects.add(cur_asp.lower())
                        row = comparative_sentences.iloc[i]
                        asp_df = asp_df.append(
                                {'object_a': row['object_a'],
                                 'object_b': row['object_b'],
                                 'aspect': cur_asp,
                                 'sentence': row['sentence'],
                                 'max': row['max'],
                                }, ignore_index=True)
            aspects = list(aspects)
            
        obj_a_aspects = []
        obj_b_aspects = []
        for aspect in aspects:
            rows = asp_df[asp_df['aspect']==aspect]
            if obj_a == rows.iloc[0]['object_a']:
                obj_a_aspects.append(aspect)
            else:
                obj_b_aspects.append(aspect)
                
        return obj_a, obj_b, obj_a_aspects, obj_b_aspects
                    
        
def request_elasticsearch(obj_a, obj_b, user, password):
    url = 'http://ltdemos.informatik.uni-hamburg.de/depcc-index/_search?q='
    url += 'text:\"{}\"%20AND%20\"{}\"'.format(obj_a, obj_b)

    size = 10000
    
    url += '&from=0&size={}'.format(size)
    response = requests.get(url, auth=HTTPBasicAuth(user, password))
    return response

def extract_sentences(es_json, aggregate_duplicates=False):
    try:
        hits = es_json.json()['hits']['hits']
    except KeyError:
        return []
    sentences = []
    seen_sentences = set()
    for hit in hits:
        source = hit['_source']
        text = source['text']

        if not aggregate_duplicates:
            if (text.lower()) not in seen_sentences:
                seen_sentences.add(text.lower())
                sentences.append(text)
        else:
            sentences.append(text)

    return sentences

def remove_questions(sentences):
    sentences_to_delete = []
    for sentence in sentences:
        if '?' in sentence:
            sentences_to_delete.append(sentence)
    for sentence in sentences_to_delete:
        del sentences[sentences.index(sentence)]
    return sentences

def get_regEx(sequence):
    return re.compile('\\b{}\\b|\\b{}\\b'.format(re.escape(sequence), re.sub('[^a-zA-Z0-9 ]', ' ', sequence)), re.IGNORECASE)

def find_pos_in_sentence(sequence, sentence):
    regEx = get_regEx(sequence)
    match = regEx.search(sentence)    
    if match == None:
        match = regEx.search(re.sub(' +',' ', re.sub('[^a-zA-Z0-9 ]', ' ', sentence)))
        return match.start() if match != None else -1
    else:
        return match.start()

def prepare_sentence_DF(sentences, obj_a, obj_b):
    index = 0
    temp_list = []
    for sentence in sentences:
        pos_a = find_pos_in_sentence(obj_a, sentence)
        pos_b = find_pos_in_sentence(obj_b, sentence)
        if pos_a < pos_b:
            temp_list.append([obj_a, obj_b, sentence])
        else:
            temp_list.append([obj_b, obj_a, sentence])
        index += 1
    sentence_df = pd.DataFrame.from_records(temp_list, columns=['object_a', 'object_b', 'sentence'])

    return sentence_df