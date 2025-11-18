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
