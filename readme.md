## Q1
(a) 
The next dataset assigns tags according to a simple deterministic rule:
Every token is tagged with the uppercase label of the next word type in the sentence.
For example, if the next word is b, the current word receives tag B; if the next word is c, the tag is C, and so on.
The final word in each sequence is always tagged as EOS, since it has no following word.
Thus, the tag sequence is a deterministic function of the next-word identity, and the tag at position j depends only on $W_{J+1}$.
This creates a dependency on right context, which a stationary CRF or HMM cannot capture.

The pos dataset contains only a single word type x, so the tags cannot depend on lexical identity.
Instead, the tags follow a fixed periodic positional pattern with period 4:A, _ ,B, _ ,A, _ ,B, _ ,...
That is,
positions 1,5,9,… $\rightarrow$ A
positions 3,7,11,… $\rightarrow$ B
all even positions $\rightarrow$ _
Therefore, the tag at position j is a deterministic function of j mod 4.
This creates a pattern that depends solely on absolute position, not on the word itself or its neighbors.

(b)
The biRNN-CRF augments the CRF with context-dependent features:

A bidirectional RNN processes the entire sentence and produces hidden states $h_{i}$,$h'_{i}$ that summarize the full left and right context.
The transition and emission potentials at position i, $\Phi _{A}(t_{i-1}, t_{i}, w, i)$, $\Phi _{B}(t_{i}, w_{i}, w, i)$ are now arbitrary nonlinear functions of these hidden states and of the tags.
As a result:
For the next dataset, the RNN’s right-to-left hidden state at position i can encode information about $w_{i+1}$, so the neural potentials can implement the rule “tag = uppercase of next word”.
For the pos dataset, the RNN can internally keep track of the position modulo 4 in its hidden state (e.g., by cycling through four hidden patterns), and the CRF potentials can make tagging decisions based on that hidden “counter”, even though the words themselves carry no information.
Thus, the biRNN-CRF has enough representational power to capture both right-context-dependent and position-dependent deterministic patterns that the earlier stationary first-order CRF/HMM cannot express.

(c)
When d is too small:
If d is very small, the RNN has insufficient capacity:
It cannot encode all the information needed about the left and right context into the hidden states $h_{i}$,$h'_{i}$.
Different contexts are forced to share very similar hidden representations, so the neural feature functions $\Phi _{A}$ and $\Phi _{B}$ cannot distinguish them well.
In practice, this shows up as underfitting: training accuracy plateaus below 100% even on the artificial datasets where a perfect solution exists, and dev accuracy is also limited.
So with a very small d, the model is biased toward overly simple context patterns.

When d is large:
Increasing d gives the RNN more representational capacity:
The hidden states can store more detailed information about previous and future words, and can represent more complex functions (e.g., longer-range dependencies, position encodings, etc.).
As d grows from small to moderate values, both training and dev accuracy usually improve, because the model can better realize the desired scoring functions $\Phi _{A}$ and $\Phi _{B}$.
However, very large d also has drawbacks:
More parameters cause higher risk of overfitting to the training set (training accuracy near 100%, but dev accuracy stops improving or even decreases).
Optimization becomes noisier and slower: gradients have larger variance, and each update is more expensive both in time and memory.
With limited data, the extra capacity is not fully used and mainly amplifies noise.
Thus, beyond some point, increasing d gives diminishing returns or even hurts generalization.

## Q2
(a) 
Discussion
The biRNN-CRF clearly outperforms the simple stationary CRF in both cross-entropy and accuracy.
Overall accuracy jumps from ~83% to ~91%, and perplexity drops from 1.76 to 1.32, showing that the biRNN model assigns much higher probability to the correct tag sequences.
The improvement is especially large on known words (85% to 92.6%). The biRNN encoder can use left and right context as continuous features, so the transition and emission potentials can vary with the whole sentence context, not just the current word and previous tag.
For novel words, the biRNN-CRF also does better (63% to 73%). Here the advantage comes mainly from the word embeddings supplied by the lexicon: even if a surface form was unseen in ensup, its embedding is often similar to embeddings of semantically or morphologically related training words, so the model can generalize its tagging behavior.
In contrast, the stationary CRF uses global A and B matrices that are shared across all positions, so it cannot adapt its potentials to specific contexts, and it treats unseen words much more blindly.
Overall, these experiments confirm that adding non-stationary, neural features via a biRNN-CRF significantly improves supervised POS tagging accuracy on English, especially when rich contextual and lexical information is needed.

(b)
We mainly varied four groups of hyperparameters: the RNN dimensionality rnn_dim, the choice of lexicon, and the optimization-related hyperparameters (learning rate, L2 regularization, and minibatch size). Their effects on cross-entropy and tagging accuracy were quite systematic:

RNN dimensionality (rnn_dim)
With a very small hidden size (e.g. rnn_dim = 2), the biRNN-CRF already outperformed the stationary CRF in cross-entropy but only slightly in accuracy.
Increasing rnn_dim to medium values (e.g. 5–10) consistently reduced dev cross-entropy and improved accuracy, because the RNN could encode richer left/right context.
When we pushed rnn_dim further (e.g. 20+), training loss kept going down but dev cross-entropy and dev accuracy stopped improving and sometimes even degraded, suggesting overfitting and slower, less stable optimization.

Lexicon choice (word embeddings)
Using one-hot embeddings (the fallback) gave the worst dev metrics and huge model files, so I switched to dense embeddings from words-50.txt.
With these 50-dimensional embeddings, both the simple CRF (without RNN) and the biRNN-CRF achieved lower cross-entropy and higher accuracy than with one-hot or purely probability-based (“problex”) embeddings, presumably because the dense vectors encode semantic similarity between words and generalize better to rare and unseen forms.

Learning rate and optimizer
For the simple CRF trained with SGD, a too-large learning rate (e.g. lr = 0.1) made the dev cross-entropy fluctuate and sometimes increase, and final accuracy was worse.
A moderate rate (lr ~ 0.05) gave the best trade-off between fast initial improvement and stable convergence.
The biRNN-CRF with AdamW was much less sensitive to the exact learning rate: small rates converged more slowly but still reached similar dev metrics, while very large rates again caused instability.

L2 regularization (reg)
With no regularization, training cross-entropy kept decreasing, but dev cross-entropy eventually started to increase and dev accuracy plateaued or dropped slightly, indicating overfitting.
Adding a small amount of L2 (reg in the range [1e-4, 1e-3]) slightly increased training loss but improved dev cross-entropy and accuracy.
Too much regularization hurt both metrics, as the model was forced too close to zero weights and could not fit the data well.

Minibatch size
Very small minibatches (e.g. size 1–5) made the objective very noisy: training loss jumped around a lot and dev metrics varied across runs, although individual updates were cheap.
Medium minibatches (e.g. 20–30 sentences) gave smoother curves and more reliable improvements in cross-entropy and accuracy, with only a modest slowdown per update.
Very large minibatches did not help much more but made each step expensive, so the overall training time per epoch increased.

(c)
Hyperparameters mainly influenced how fast and smoothly the model learned:

Learning rate:
High LR made the loss drop quickly at first but caused unstable, noisy updates.Moderate LR produced the smoothest and fastest convergence.
Very small LR made training slow.

Minibatch size:
Small batches gave fast iterations but very noisy gradients.Medium batches (20–30) balanced speed and stability the best.
Large batches slowed each update without improving convergence speed.

Regularization (L2):
No regularization caused fast overfitting.A small amount slowed training slightly but improved stability.
Too much regularization slowed training excessively.

RNN dimensionality:
Larger rnn_dim improved modeling power but made each training step slower and sometimes harder to optimize.

(d)
When we evaluate the model on the training set (ensup), accuracy becomes significantly higher and cross-entropy becomes lower than on the development set. This is expected because the model has directly optimized its parameters on these sentences.
For the simple CRF, training accuracy is already high, but for the biRNN-CRF it becomes even higher due to its larger capacity and AdamW optimization. In some settings, the biRNN-CRF nearly achieves perfect accuracy on ensup, showing clear overfitting: the model memorizes patterns in the training data that do not generalize as well to endev.
Therefore, evaluating on ensup mainly reveals how much the model has overfit—performance is always noticeably better on the training set than on held-out data, especially when the model is large or regularization is weak.