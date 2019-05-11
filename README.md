# Transformer Model Encoder
* Predicting next chars/words without time-based RNN variants, instead using Attention towards previous chars/words,thence predicting next chars/words, based on previous chars/words.
### Observations:
1. Say target string is "was only be", there are 2 spaces in it, and say at some iteration i, output is some gibberish(say,"wlllsssabs"), then the loss corresponding to these two spaces will start decreasing at nearly the same iteration, clearly showing some sort of learnt connection among space characters. That means the network is learning somekind of attentive dependancy for space characters and similarly for others.

*Check the closed issue!!
