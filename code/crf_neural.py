#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

           # you fill this in!
        d = self.rnn_dim
        e = self.e
        k = self.k      
        # h_j   = σ( M  [1; h_{j-1}; w_j] )      (forward)
        # h'_i  = σ( M' [1; w_i; h'_{i+1}] )     (backward)
        self.M = nn.Parameter(torch.empty(d, 1 + d + e))
        self.M_prime = nn.Parameter(torch.empty(d, 1 + e + d))
        nn.init.xavier_uniform_(self.M)
        nn.init.xavier_uniform_(self.M_prime)

        # f_A(s,t,w,i) = σ( U_A [1; h_{i-2}; s_vec; t_vec; h'_i] )  transition features
        fA_dim = d
        self.U_A = nn.Parameter(torch.empty(fA_dim, 1 + d + k + k + d))
        nn.init.xavier_uniform_(self.U_A)
        self.theta_A = nn.Parameter(torch.empty(fA_dim))
        nn.init.normal_(self.theta_A, mean=0.0, std=0.1)

        # f_B(t,w,w,i) = σ( U_B [1; h_{i-1}; t_vec; w_vec; h'_i] ) emission features
        fB_dim = d
        self.U_B = nn.Parameter(torch.empty(fB_dim, 1 + d + k + e + d))
        nn.init.xavier_uniform_(self.U_B)
        self.theta_B = nn.Parameter(torch.empty(fB_dim))    
        nn.init.normal_(self.theta_B, mean=0.0, std=0.1)



        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        # you fill this in!
        device = self.M.device
        # get word id
        if isinstance(isent.words, Tensor):
            word_ids = isent.words.to(device=device, dtype=torch.long)
        else:
            word_ids = torch.tensor(isent.words, dtype=torch.long, device=device)

        self.word_ids = word_ids
        n = word_ids.size(0)
        self.sent_len = n

        #word embeddings
        E = self.E.to(device=device)
        w_embeds = E[word_ids]  # (n, e)

        d = self.rnn_dim

        # forward RNN
        # h_fwd[i] represents first i words' prefix, h_fwd[0] = theta
        h_fwd_list = []
        h_prev = torch.zeros(d, device=device)
        h_fwd_list.append(h_prev) 
        for i in range(n):
            x_i = w_embeds[i]  # (e,)
            inp = torch.cat([torch.ones(1, device=device), h_prev, x_i], dim=0)  # (1 + d + e,)
            h_curr = torch.sigmoid(self.M @ inp)  # (d,)
            h_fwd_list.append(h_curr)
            h_prev = h_curr

        self.h_fwd = torch.stack(h_fwd_list, dim=0)  # (n+1, d)
        # backward RNN
        # h_bwd[i] represents suffix starting from i, h_bwd[n] = theta
        h_bwd_list = []
        h_next = torch.zeros(d, device=device)
        h_bwd_list.append(h_next)
        for i in reversed(range(n)):
            x_i = w_embeds[i]  # (e,)
            inp = torch.cat([torch.ones(1, device=device), x_i, h_next], dim=0)  # (1 + e + d,)
            h_curr = torch.sigmoid(self.M_prime @ inp)  # (d,)
            h_bwd_list.append(h_curr)
            h_next = h_curr

        self.h_bwd = torch.stack(list(reversed(h_bwd_list)), dim=0)  # (n+1, d)

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

         # you fill this in!
        #  f_A(s,t,w,i) = σ( U_A [1; h_{i-2}; s; t; h'_i] )
        #   i = position
        #   h_{i-2} ~= h_fwd[max(i-1, 0)]
        #   h'_i    ~= h_bwd[min(i, n)]
        assert self._h_fwd is not None and self._h_bwd is not None, \
            "setup_sentence must be called before A_at"
        assert self._sent_len is not None
        n = self._sent_len
        k = self.k
        device = self.M.device

        i = position

        # h_{i-2} approx：max(i-1, 0) prefix
        idx_left = max(i - 1, 0)
        h_left = self._h_fwd[idx_left]            # (d,)
        # h'_i approx：from i suffix
        idx_right = min(i, n)
        h_right = self._h_bwd[idx_right]          # (d,)

        # tag one-hot: (k, k)
        tag_eye = self.tag_eye.to(device=device)

        # key：generate input for all (s,t) pairs
        # shape (k*k, F_A) matrix, each row corresponds to a (s,t) pair

        # s_vecs: (k*k, k) Repeat the row index s, and assign all t under each s
        s_vecs = tag_eye.unsqueeze(1).expand(k, k, k).reshape(-1, k)   # (k*k, k)
        # t_vecs: (k*k, k) Repeat the column index t, and assign all s under each t
        t_vecs = tag_eye.unsqueeze(0).expand(k, k, k).reshape(-1, k)   # (k*k, k)

        # h_left, h_right: (k*k, d)
        h_left_rep = h_left.unsqueeze(0).expand(k * k, -1)
        h_right_rep = h_right.unsqueeze(0).expand(k * k, -1)

        # ones: (k*k, 1)
        ones = torch.ones(k * k, 1, device=device)

        # inp_all: (k*k, 1 + d + k + k + d)
        inp_all = torch.cat(
            [ones, h_left_rep, s_vecs, t_vecs, h_right_rep],
            dim=1
        )

        # f_A_all: (k*k, d)
        f_A_all = torch.sigmoid(inp_all @ self.U_A.T)

        # score_all: (k*k,)
        score_all = f_A_all @ self.theta_A

        # ϕA(s,t) reshape (k, k)
        A = torch.exp(score_all).reshape(k, k)

        return A
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        # you fill this in!
        assert self._h_fwd is not None and self._h_bwd is not None and self._word_ids is not None, \
            "setup_sentence must be called before B_at"
        assert self._sent_len is not None

        n = self._sent_len
        k = self.k
        V = self.V
        device = self.M.device

        i = position
        w_id = int(self._word_ids[i].item())

        # context vectors
        h_left = self._h_fwd[i]                # (d,)
        idx_right = min(i, n)
        h_right = self._h_bwd[idx_right]       # (d,)

        # current word embedding
        E = self.E.to(device)
        w_vec = E[w_id]                        # (e,)

        # tag one-hot: (k, k)
        tag_eye = self.tag_eye.to(device=device)

        # Construct input matrix for all tags at once (k, F_B) 
        # 1: (k, 1)
        ones = torch.ones(k, 1, device=device)
        # h_left: (k, d)
        h_left_exp = h_left.unsqueeze(0).expand(k, -1)
        # t_vec: (k, k) 
        t_mat = tag_eye                        # (k, k)
        # w_vec: (k, e)
        w_vec_exp = w_vec.unsqueeze(0).expand(k, -1)
        # h_right: (k, d)
        h_right_exp = h_right.unsqueeze(0).expand(k, -1)

        # inp: (k, 1 + d + k + e + d)
        inp = torch.cat(
            [ones, h_left_exp, t_mat, w_vec_exp, h_right_exp],
            dim=1
        )

        # f_B = σ(U_B @ inp^T)，
        # U_B: (d, F_B) → inp @ U_B^T: (k, d)
        f_B = torch.sigmoid(inp @ self.U_B.T)      # (k, d)

        # score = θ_B · f_B，dot product by row
        score = f_B @ self.theta_B                 # (k,)

        # potential ϕB(t, w_i)
        phi = torch.exp(score)                     # (k,)

        # consruct B(j): (k, V)，only the w_id column is useful
        B = torch.ones(k, V, device=device)
        B[:, w_id] = phi

        return B
