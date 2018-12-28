# Attention-recurrence-encoder
* Predicting next chars/words without using RecurrentNeuralNets, instead using Attention towards previous chars/words,thence predicting next chars/words, based on previous chars/words.
* This one is character-based.
* To run it, just use a text corpus, name it seqText.txt(or change the name inside the python script).
### Something peculiar:
Say target string is "was only be", there are 2 spaces in it, and say at some iteration i, output is some gibberish(say,"wlllsssabs"), then the spaces will appear at its 2 respective places on the same iteration,clearly showing that every space character is linked, that means the network is learning somekind of attentive dependancy for space characters and similarly for others.
