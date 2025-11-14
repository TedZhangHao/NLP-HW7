#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle
import collections
from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors
        self.alpha = None
        self.log_A = None
        self.log_B = None
        self.A_counts = None
        self.B_counts = None
        self.log_Z = None
        # self.A_contrib_log = []
        # self.B_contrib_log = []
        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###
        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # Update emission probabilities (self.B).
        self.B_counts += λ          # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
        self.B[self.eos_t, :] = 0   # replace these nan values with structural zeroes, just as in init_params
        self.B[self.bos_t, :] = 0

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'
                
        # Update transition probabilities (self.A).  
        # Don't forget to respect the settings self.unigram and λ.
        # See the init_params() method for a discussion of self.A in the
        # unigram case.
        if not self.unigram:
            self.A_counts += λ      # smooth the counts
            self.A[:, self.bos_t] = 0
            self.A[self.eos_t, :]= 0
            self.A[self.bos_t, self.eos_t] = 0
            self.A = self.A_counts / self.A_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
            self.A[:, self.bos_t] = 0
            self.A[self.eos_t, :] = 0
            self.A[self.bos_t, self.eos_t] = 0
        else:
            # unigram HMM (0th-order)
            # Treat A as tag priors p_A(t_i), independent of previous tag.
            # So we ignore A_counts[i, j] transitions, and only use tag frequency.
            tag_totals = self.A_counts.sum(dim=1) + λ  # smooth each tag
            tag_totals[self.bos_t] = 0  # BOS, EOS are structural zeroes
            tag_totals[self.eos_t] = 0

            # Normalize to make it a proper distribution p_A(t_i)
            self.A = tag_totals / tag_totals.sum()

            # For consistency with bigram shape, make A a matrix with identical rows
            self.A = self.A.unsqueeze(0).repeat(self.k, 1)

            # Structural zero enforcement
            self.A[:, self.bos_t] = 0
            self.A[self.eos_t, :] = 0
            self.A[self.bos_t, self.eos_t] = 0

    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.005,
              max_steps: int = 50000,
              save_path: Optional[Path|str] = "my_hmm.pkl") -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      

        self._save_time = time.time()      # mark start of training     
        dev_loss = loss(self)              # evaluate the model at the start of training
        old_dev_loss: float = dev_loss     # loss from the last epoch
        steps: int = 0   # total number of sentences the model has been trained on so far      
        while steps < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            if save_path: self.save(save_path, checkpoint=steps)  # save incompletely trained model in case we crash
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # Save the trained model.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        # Integerize the words and tags of the given sentence, which came from the given corpus.
        isent = self._integerize_sentence(sentence, corpus)
        return self.forward_pass(isent)

    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Forward-backward algorithm.
        log_Z_forward = self.forward_pass(isent)
        log_Z_backward = self.backward_pass(isent, mult=mult)
        
        # Check that forward and backward passes found the same total
        # probability of all paths (up to floating-point error).
        assert torch.isclose(log_Z_forward, log_Z_backward), f"backward log-probability {log_Z_backward} doesn't match forward log-probability {log_Z_forward}!"


    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        # The "nice" way to construct the sequence of vectors alpha[0],
        # alpha[1], ...  is by appending to a List[Tensor] at each step.
        # But to better match the notation in the handout, we'll instead
        # preallocate a list alpha of length n+2 so that we can assign 
        # directly to each alpha[j] in turn.
        alpha = [torch.empty(self.k) for _ in isent]    
        alpha[0] = self.eye[self.bos_t]  # vector that is one-hot at BOS_TAG

            # Note: once you have this working on the ice cream data, you may
            # have to modify this design slightly to avoid underflow on the
            # English tagging data. See section C in the reading handout.
        log_A = torch.log(self.A+1e-12)
        log_B = torch.log(self.B+1e-12)
        self.log_A = log_A
        self.log_B = log_B
        n = len(isent)
        logalpha = [torch.full((self.k,), -float("inf")) for _ in range(n)]
        logalpha[0] = torch.log(alpha[0])
        # k = [0.0 for _ in isent] 
        
        eos_w = self.V       # EOS_WORD id
        bos_w = self.V + 1   # BOS_WORD id
        for j in range(1, n):
            word_id, tag_id = isent[j]
            logalpha[j] = torch.logsumexp(logalpha[j-1].unsqueeze(1) + log_A, dim=0) 
            if word_id < self.V :
                log_emiss = log_B[:, word_id]
            else:
                log_emiss = torch.full((self.k,), -float("inf"), device=log_B.device)
                if word_id == eos_w:
                    log_emiss[self.eos_t] = 0.0
                elif word_id == bos_w:
                    log_emiss[self.bos_t] = 0.0

            logalpha[j] += log_emiss
            if tag_id is not None and tag_id >= 0:
                mask = torch.full((self.k,), -float("inf"))
                mask[tag_id] = 0.0
                logalpha[j] += mask

        self.alpha = logalpha  # remember for backward pass
        log_Z = torch.logsumexp(logalpha[-1], dim=0)
        self.log_Z = log_Z      # remember for backward pass
        return log_Z

    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the backward
        probability). 
        
        As a side effect, add the expected transition and emission counts (times
        mult) into self.A_counts and self.B_counts.  These depend on the alpha
        values and log Z, which were stored for us (in self) by the forward
        pass."""

        # Pre-allocate beta just as we pre-allocated alpha.
        beta = [torch.empty(self.k) for _ in isent]
        beta[-1] = self.eye[self.eos_t]  # vector that is one-hot at EOS_TAG
        n = len(isent)
        eos_w = self.V
        bos_w = self.V + 1
        logbeta = [torch.full((self.k,), -float("inf")) for _ in range(n)]
        logbeta[-1] = torch.log(beta[-1])
         # reversed recursion j = n-2, ..., 0
        for j in reversed(range(n - 1)):
            word_id_next, tag_id_next = isent[j + 1]

            # emission log prob vector
            if word_id_next < self.V:
                emit_next = self.log_B[:, word_id_next]
            else:
                emit_next = torch.full((self.k,), -float("inf"))
                if word_id_next == eos_w:
                    emit_next[self.eos_t] = 0.0
                elif word_id_next == bos_w:
                    emit_next[self.bos_t] = 0.0

            
            inner = self.log_A + emit_next.unsqueeze(0) + logbeta[j + 1].unsqueeze(0)  # (k,k)
            logbeta[j] = torch.logsumexp(inner, dim=1)  # (k,)

            # consideing supervised tag at position j
            _, tag_j = isent[j]
            if tag_j is not None and tag_j >= 0:
                mask = torch.full((self.k,), -float("inf"))
                mask[tag_j] = 0.0
                logbeta[j] = logbeta[j] + mask

        self._beta = logbeta   # remember for expected counts
        # backward 
        log_Z_backward = torch.logsumexp(self.alpha[-1], dim=0)

        # calculate expected counts
        alpha = self.alpha
        logZ = self.log_Z

        # emission expected counts
        for j in range(1, n - 1):            # omit BOS and EOS positions
            word_id, _ = isent[j]
            if word_id >= self.V:
                continue                     # no columns for BOS/EOS words

            log_gamma = alpha[j] + logbeta[j] - logZ
            gamma = torch.exp(log_gamma)

            # no emissions from BOS/EOS tags
            gamma[self.bos_t] = 0.0
            gamma[self.eos_t] = 0.0

            self.B_counts[:, word_id] += mult * gamma

        # transition expected counts
        for j in range(0, n - 1):
            word_id_next, _ = isent[j + 1]

            if word_id_next < self.V:
                emit_next = self.log_B[:, word_id_next]
            else:
                emit_next = torch.full((self.k,), -float("inf"))
                if word_id_next == eos_w:
                    emit_next[self.eos_t] = 0.0
                elif word_id_next == bos_w:
                    emit_next[self.bos_t] = 0.0

            xi_log = (
                alpha[j].unsqueeze(1)        # (k,1)
                + self.log_A                      # (k,k)
                + emit_next.unsqueeze(0)     # (1,k)
                + logbeta[j + 1].unsqueeze(0)
                - logZ
            )
            xi = torch.exp(xi_log)
            # if hasattr(self, "A_contrib_log"):
            #     # save fractional transition counts
            #     for s in range(self.k):
            #         for t in range(self.k):
            #             contrib = xi[s, t].item()
            #             if contrib > 1e-5:
            #                 prev_word = self.vocab[isent[j][0]]
            #                 next_word = self.vocab[isent[j + 1][0]]
            #                 self.A_contrib_log.append((prev_word, next_word, s, t, contrib))
            # if hasattr(self, "B_contrib_log"):
            #     for j in range(1, n - 1):
            #         word_id, _ = isent[j]
            #         if word_id < self.V:
            #             gamma = torch.exp(alpha[j] + beta[j] - logZ)
            #             for t in range(self.k):
            #                 if gamma[t] > 1e-5:
            #                     self.B_contrib_log.append((self.vocab[word_id], self.tagset[t], gamma[t].item()))


            # no transitions to BOS, from EOS
            xi[:, self.bos_t] = 0.0
            xi[self.eos_t, :] = 0.0

            self.A_counts += mult * xi

        assert torch.isclose(log_Z_backward, logZ, atol=1e-3)
        return log_Z_backward
    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # The code continues to use the name alpha, rather than \hat{alpha}
        # as in the handout.

        # We'll start by integerizing the input Sentence. You'll have to
        # deintegerize the words and tags again when constructing the return
        # value, since the type annotation on this method says that it returns a
        # Sentence object, and that's what downstream methods like eval_tagging
        # will expect.  (Running mypy on your code will check that your code
        # conforms to the type annotations ...)

        isent = self._integerize_sentence(sentence, corpus)
        n = len(isent)
        # See comments in log_forward on preallocation of alpha.
        alpha        = [torch.empty(self.k)                  for _ in isent]  
        backpointers = [torch.empty(self.k, dtype=torch.int) for _ in isent]
        tags: List[int]    # you'll put your integerized tagging here

        alpha[0] = self.eye[self.bos_t]  # vector that is one-hot at BOS_TAG

            # Note: once you have this working on the ice cream data, you may
            # have to modify this design slightly to avoid underflow on the
            # English tagging data. See section C in the reading handout.
        log_A = torch.log(self.A+1e-12)
        log_B = torch.log(self.B+1e-12)
        self.log_A = log_A
        self.log_B = log_B
        logalpha = [torch.full((self.k,), -float("inf")) for _ in range(n)]
        logalpha[0] = torch.log(alpha[0])
        # k = [0.0 for _ in isent] 
        eos_w, bos_w = self.V, self.V + 1
        for j in range(1, n):
            word_id, tag_id = isent[j]
            temp = logalpha[j-1].unsqueeze(1) + log_A 
            logalpha[j], backpointers[j] = torch.max(temp, dim=0)
            # print(" isent[j]:",isent[j])
            # backpointers[j] = torch.argmax(temp, dim=0)
            if word_id < self.V:
                log_emiss = log_B[:, word_id]
            else:
                log_emiss = torch.full((self.k,), -float("inf"), device=log_B.device)
                if word_id == eos_w:
                    log_emiss[self.eos_t] = 0.0
                elif word_id == bos_w:
                    log_emiss[self.bos_t] = 0.0
            logalpha[j] += log_emiss
        # temp = logalpha[n-2].unsqueeze(1) + log_A[:, self.eos_t]
        # logalpha[-1], backpointers[-1] = torch.max(temp, dim=0)
            
            # k[j] = torch.max(alpha[j])
        tags = [0 for _ in range(n)]
        # alpha[-1] = self.eye[self.eos_t]
        tags[-1] = self.eos_t  #torch.argmax(alpha[-1]).item()
        for j in range(n-1, 0, -1):
                tags[j-1] = backpointers[j][tags[j]].item()

        # Make a new tagged sentence with the old words and the chosen tags
        # (using self.tagset to deintegerize the chosen tags).
        return Sentence([(word, self.tagset[tags[j]]) for j, (word, tag) in enumerate(sentence)])

    def save(self, path: Path|str, checkpoint=None, checkpoint_interval: int = 300) -> None:
        """Save this model to the file named by path.  Or if checkpoint is not None, insert its 
        string representation into the filename and save to a temporary checkpoint file (but only 
        do this save if it's been at least checkpoint_interval seconds since the last save).  If 
        the save is successful, then remove the previous checkpoint file, if any."""

        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed

        now = time.time()
        old_save_time =           getattr(self, "_save_time", None)
        old_checkpoint_path =     getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None   # this is a real save, not a checkpoint
        else:    
            if old_save_time is not None and now < old_save_time + checkpoint_interval: 
                return   # we already saved too recently to save another temp version
            path = path.with_name(f"{path.stem}-{checkpoint}{path.suffix}")  # use temp filename
            self._checkpoint_path = path

        # update the elapsed training time since we started training or last saved (if that happened)
        if old_save_time is not None:
            self.total_training_time = old_total_training_time + (now - old_save_time)
        del self._save_time
        
        # Save the model with the fields set as above, so that we'll 
        # continue from it correctly when we reload it.
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            # something went wrong with the save; so restore our old fields,
            # so that caller can potentially catch this exception and try again
            self._save_time          = old_save_time
            self._checkpoint_path    = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e
        
        # Since save was successful, remember it and remove old temp version (if any)
        self._save_time = now
        if old_checkpoint_path: 
            try: os.remove(old_checkpoint_path)
            except FileNotFoundError: pass  # don't complain if the user already removed it manually

    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model
