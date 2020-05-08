import torch
import torch.nn as nn

class CRF(nn.Module):
    def __init__(self, states_num, pad_idx, start_idx, device):
        super(CRF, self).__init__()
        self.states_num = states_num
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        # Transition matrix contains log probabilities from state j to state i
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num, dtype=torch.float))
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        # Default initialization
        self.transition_matrix.data[self.start_idx, :] = -9999.0
        self.transition_matrix.data[:, self.pad_idx] = -9999.0
        self.transition_matrix.data[self.pad_idx, :] = -9999.0
        self.transition_matrix.data[self.pad_idx, self.pad_idx] = 0.0

    def get_empirical_transition_matrix(self, tag_sequences_train, tag2ix):

        empirical_transition_matrix = torch.zeros(self.states_num, self.states_num, dtype=torch.long)
        for tag_seq in tag_sequences_train:
            s = tag2ix[tag_seq[0]]
            empirical_transition_matrix[s, self.start_idx] += 1
            for n, tag in enumerate(tag_seq):
                if n + 1 >= len(tag_seq):
                    break
                next_tag = tag_seq[n + 1]
                j = tag2ix[tag]
                i = tag2ix[next_tag]
                empirical_transition_matrix[i, j] += 1
        return empirical_transition_matrix

    def init_transition_matrix_empirical(self, tag_sequences_train, tag2ix):
        print('Initializing transition matrix.')
        # Calculate statistics for tag transitions
        empirical_transition_matrix = self.get_empirical_transition_matrix(tag_sequences_train, tag2ix)
        # Initialize
        for i in range(len(tag2ix)):
            for j in range(len(tag2ix)):
                if empirical_transition_matrix[i, j] == 0:
                    self.transition_matrix.data[i, j] = -9999.0
                #self.transition_matrix.data[i, j] = torch.log(empirical_transition_matrix[i, j].float() + 10**-32)
        print('Initialized transition matrix.')

    def numerator(self, features_rnn_compressed, states_tensor, mask_tensor, device):
        # features_input_tensor: batch_num x max_seq_len x states_num
        # states_tensor: batch_num x max_seq_len
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = states_tensor.shape
        score = torch.zeros(batch_num, dtype=torch.float).to(device)
        start_states_tensor = torch.zeros(batch_num, 1, dtype=torch.long).fill_(self.start_idx).to(device)
        states_tensor = torch.cat([start_states_tensor, states_tensor], 1)
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n]
            curr_emission = torch.zeros(batch_num, dtype=torch.float).to(device)
            curr_transition = torch.zeros(batch_num, dtype=torch.float).to(device)
            for k in range(batch_num):
                curr_emission[k] = features_rnn_compressed[k, n, states_tensor[k, n + 1]].unsqueeze(0)
                curr_states_seq = states_tensor[k]
                # !!!
                curr_transition[k] = self.transition_matrix[curr_states_seq[n + 1], curr_states_seq[n]].unsqueeze(0)
            score = score + curr_emission*curr_mask + curr_transition*curr_mask
        return score

    def denominator(self, features_rnn_compressed, mask_tensor, device):
        # features_rnn_compressed: batch x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = mask_tensor.shape
        score = torch.zeros(batch_num, self.states_num, dtype=torch.float).fill_(-9999.0).to(device)
        score[:, self.start_idx] = 0.
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n].unsqueeze(-1).expand_as(score)
            curr_score = score.unsqueeze(1).expand(-1, *self.transition_matrix.size())
            curr_emission = features_rnn_compressed[:, n].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transition_matrix.unsqueeze(0).expand_as(curr_score)
            #curr_score = torch.logsumexp(curr_score + curr_emission + curr_transition, dim=2)
            curr_score = log_sum_exp(curr_score + curr_emission + curr_transition)
            score = curr_score * curr_mask + score * (1 - curr_mask)
        #score = torch.logsumexp(score, dim=1)
        score = log_sum_exp(score)
        return score

    def decode_viterbi(self, features_rnn_compressed, mask_tensor, device):
        # features_rnn_compressed: batch x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        batch_size, max_seq_len = mask_tensor.shape
        seq_len_list = [int(mask_tensor[k].sum().item()) for k in range(batch_size)]
        # Step 1. Calculate scores & backpointers
        score = torch.Tensor(batch_size, self.states_num).fill_(-9999.).to(device)
        score[:, self.start_idx] = 0.0
        backpointers = torch.LongTensor(batch_size, max_seq_len, self.states_num).to(device)
        for n in range(max_seq_len):
            curr_emissions = features_rnn_compressed[:, n]
            curr_score = torch.Tensor(batch_size, self.states_num).to(device)
            curr_backpointers = torch.LongTensor(batch_size, self.states_num).to(device)
            for curr_state in range(self.states_num):
                T = self.transition_matrix[curr_state, :].unsqueeze(0).expand(batch_size, self.states_num)
                max_values, max_indices = torch.max(score + T, 1)
                curr_score[:, curr_state] = max_values
                curr_backpointers[:, curr_state] = max_indices
            curr_mask = mask_tensor[:, n].unsqueeze(1).expand(batch_size, self.states_num)
            score = score * (1 - curr_mask) + (curr_score + curr_emissions) * curr_mask
            backpointers[:, n, :] = curr_backpointers # shape: batch_size x max_seq_len x state_num
        best_score_batch, last_best_state_batch = torch.max(score, 1)
        # Step 2. Find the best path
        best_path_batch = [[state] for state in last_best_state_batch.tolist()]
        for k in range(batch_size):
            curr_best_state = last_best_state_batch[k]
            curr_seq_len = seq_len_list[k]
            for n in reversed(range(1, curr_seq_len)):
                curr_best_state = backpointers[k, n, curr_best_state].item()
                best_path_batch[k].insert(0, curr_best_state)
        return best_path_batch

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))