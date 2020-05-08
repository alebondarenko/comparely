import torch
import torch.nn as nn
from seq_lab.LSTMCRF_seq_lab.CRF import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

class LSTMCRFTagger(nn.Module):

    def __init__(self, embeddings, hidden_dim, tagset_size, device, lstm_layer=1, dropout_ratio=0.5):
        super(LSTMCRFTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        # load pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        # embeddings are not fine-tuned
        self.embedding.weight.requires_grad = False
        
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        
        # RNN layer with LSTM cells
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True,
                            batch_first=True)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size + 1)
        
        self.crf = CRF(tagset_size + 1, 0, tagset_size, device)

    def forward_birnn(self, sents, seq_lens, device):
        
        _, max_seq_len = sents.shape

        embeds = self.embedding(sents)
        
        embeds_d = self.dropout(embeds)
        
        embeds_packed, reverse_sort_index = self.pack(embeds_d, seq_lens, device)
        
        lstm_out_packed, _ = self.lstm(embeds_packed)
        
        lstm_out_unpacked = self.unpack(lstm_out_packed, max_seq_len, reverse_sort_index)
        
        lstm_out_unpacked_d = self.dropout(lstm_out_unpacked)

        tag_space = self.hidden2tag(lstm_out_unpacked_d)

        return tag_space
    
    def get_mask_from_lens(self, seq_lens, device):
        batch_num = len(seq_lens)
        max_seq_len = max(seq_lens)
        mask_tensor = torch.zeros(batch_num, max_seq_len, dtype=torch.float).to(device)
        for k, seq_len in enumerate(seq_lens):
            mask_tensor[k, :seq_len] = 1
        return mask_tensor
    
    def viterbi_loss(self, features_rnn, seq_lens, tags_batch, device):
        
        mask = self.get_mask_from_lens(seq_lens, device)  # batch_num x max_seq_len
        
        numerator = self.crf.numerator(features_rnn, tags_batch, mask, device)
        
        denominator = self.crf.denominator(features_rnn, mask, device)
        
        loss = -torch.mean(numerator - denominator)
        
        return loss
    
    def sort_by_seq_len_list(self, seq_len_list, device):
        data_num = len(seq_len_list)
        sort_indices = sorted(range(len(seq_len_list)), key=seq_len_list.__getitem__, reverse=True)
        reverse_sort_indices = [-1 for _ in range(data_num)]
        for i in range(data_num):
            reverse_sort_indices[sort_indices[i]] = i
        sort_index = torch.tensor(sort_indices, dtype=torch.long).to(device)
        reverse_sort_index = torch.tensor(reverse_sort_indices, dtype=torch.long).to(device)
        return sorted(seq_len_list, reverse=True), sort_index, reverse_sort_index
    
    def pack(self, input_tensor, seq_len_list, device):
        sorted_seq_len_list, sort_index, reverse_sort_index = self.sort_by_seq_len_list(seq_len_list, device)
        input_tensor_sorted = torch.index_select(input_tensor, dim=0, index=sort_index)
        return pack_padded_sequence(input_tensor_sorted, lengths=sorted_seq_len_list, batch_first=True), \
               reverse_sort_index

    def unpack(self, output_packed, max_seq_len, reverse_sort_index):
        output_tensor_sorted, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=max_seq_len)
        output_tensor = torch.index_select(output_tensor_sorted, dim=0, index=reverse_sort_index)
        return output_tensor