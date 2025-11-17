#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with some arbitrary non-stationary features, for testing."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        # an __init__() call to the nn.Module class must be made before assignment on the child.
        nn.Module.__init__(self)  

        self.E = lexicon          # rows are word embeddings
        self.e = lexicon.size(1)  # dimensionality of word embeddings
        self.rnn_dim = rnn_dim

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        # [docstring will be inherited from parent method]

         # you fill this in!
         # Create the stationary A and B used by parent class (init to random)
        super().init_params()

        # A simple linear layer to create tag bias from (w_j, w_{j+1})
        # input: 2e  to output: k tags
        self.context2tag = nn.Linear(2 * self.e, self.k)

    @override
    def updateAB(self) -> None:
        # Your non-stationary A_at() and B_at() might not make any use of the
        # stationary A and B matrices computed by the parent.  So we override
        # the parent so that we won't waste time computing self.A, self.B.
        #
        # But if you decide that you want A_at() and B() at to refer to self.A
        # and self.B (for example, multiplying stationary and non-stationary
        # potentials), then you'll still need to compute them; in that case,
        # don't override the parent in this way.
        pass   # do nothing

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        # [docstring will be inherited from parent method]

        # You need to override this function to compute your non-stationary features.

          # you fill this in!

        return self.A  # example
        
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:       
        # [docstring will be inherited from parent method]

        # you fill this in!

         """
        Non-stationary emission potentials.

        We modify B_j(:, w_j) based on the embeddings of:
            - current word w_j
            - next word w_{j+1}
        while keeping all other columns stationary.
        """
         device = self.A.device
         base_B = super().B_at(position, sentence).clone()   # (k, V)
         V = base_B.size(1)

         # current word id
         word_id, _ = sentence[position]

         # embedding lookup 
         E = self.E.to(device)

         # embedding of w_j
         if 0 <= word_id < E.size(0):
              e_curr = E[word_id]
         else:
              e_curr = torch.zeros(self.e, device=device)

         # embedding of w_{j+1}
         if position + 1 < len(sentence):
             next_id, _ = sentence[position + 1]
             if 0 <= next_id < E.size(0):
                 e_next = E[next_id]
             else:
                 e_next = torch.zeros(self.e, device=device)
         else:
             e_next = torch.zeros(self.e, device=device)

         # compute tag bias
         ctx = torch.cat([e_curr, e_next], dim=0).unsqueeze(0)   # (1, 2e)
         tag_bias = self.context2tag(ctx).squeeze(0)             # (k,)

         # apply bias only to column w_j
         if 0 <= word_id < V:
            col = base_B[:, word_id] + 1e-12      # avoid log(0)
            log_col = torch.log(col)
            log_col = log_col + tag_bias
            base_B[:, word_id] = torch.exp(log_col)

         return base_B
