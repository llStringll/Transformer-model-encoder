# Transformer Model Encoder
* Genrating vectors for tokens without time-based RNN variants, instead using bidirectional Attention towards contextual tokens
* This is not the full transformer model, it is only the encoder part of it, to study attention(self, masked and multi-head) among the inputs.

### Dependencies required:
* numpy (for matrices and N-dim arrays)
* matplotlib (for plotting loss)
* sys (for control over the flow during unit testing and debugging)
* time (self explanatory)
* warnings (for unit testing and debugging)

### Observations:
1. Say target string is "was only be", there are 2 spaces in it, and say at some iteration i, output is some gibberish(say,"wlllsssabs"), then the loss corresponding to these two spaces will start decreasing at nearly the same iteration, clearly showing some sort of learnt connection among space characters. That means the network is learning somekind of attentive dependancy for space characters and similarly for others.

*Check the closed issue!!
