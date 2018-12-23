import numpy as np
import sys

################################################################################
# PARSING OF DATASET

corpus=open("seqText.txt",'r').read()[0:10] # max 5M
chars=list(set(corpus))
emb=dict()
X,Y=[],[]
dmodel = len(chars)
for i in range(dmodel):
    emb[chars[i]]=np.zeros([dmodel])
    emb[chars[i]][i]=1

seq_size=5
beg=0
end=seq_size
def get_seq(): # generate consecutive overlapped sequqnces of fixed length
    global beg,end
    if end>=len(corpus)-1:
        beg=0
        end=seq_size
    inp_seq, out_seq=[],[]
    for i in range(beg,end):
        inp_seq.append(emb[corpus[i]])
        out_seq.append(emb[corpus[i+1]])
    beg+=1
    end+=1
    return np.array(inp_seq),np.array(out_seq)

print "Sequence size :",seq_size
print "single Embedding : 1 X",len(chars)
print "single Sequence matrix input :",seq_size,"X",len(chars)

################################################################################
# NETWORK ARCHITECTURE

learningRate = 0.0005 # learning rate
gamma = 0.6 # for label smoothing using mass redistribution
headSize = 3 # dmodel/headSize must be int

class LayerNorm:
    def __init__(self):
        self.gain = np.random.randn(seq_size, dmodel)
        self.bias = np.random.randn(seq_size, dmodel)

    def flow(self,x):
        self.inp = x
        self.mean = np.array([np.sum(self.inp,axis=1)/self.inp.shape[1]]).T
        self.variance = np.sqrt(np.array([np.sum((self.inp-self.mean)**2,axis=1)]).T/self.inp.shape[1])
        self.y = (self.gain/self.variance)*(self.inp-self.mean) + self.bias
        return self.y

    def backprop(self,derivOutput):
        dgain = derivOutput * (self.inp - self.mean)/self.variance
        dbias = derivOutput
        dinp = self.gain*( self.variance*((dmodel-1)/dmodel) - (self.inp-self.mean)*(self.inp-self.mean-(np.sum(self.inp-self.mean)/dmodel)/(self.variance*dmodel)) )/(self.variance**2)
        # apply gradient descent
        np.clip(dgain,-5,5,out=dgain)
        np.clip(dbias,-5,5,out=dbias)
        self.gain = self.gain - learningRate*dgain
        self.bias = self.bias - learningRate*dbias
        return dinp

class MultiheadAttentionLayer:
    def __init__(self,headsize):
        self.h = headsize
        self.Wq,self.Wk,self.Wv,self.AttLayer=[],[],[],[]
        for hitr in range(self.h):
            self.Wq.append(np.random.randn(dmodel, dmodel/self.h))
            self.Wk.append(np.random.randn(dmodel, dmodel/self.h))
            self.Wv.append(np.random.randn(dmodel, dmodel/self.h))
            self.AttLayer.append(EncoderOnlyMaskedAttentionLayer())
        self.Wo = np.random.randn(dmodel, dmodel)

    def flow(self,q,k,v):
        self.q = q
        self.k = k
        self.v = v
        self.ConcatHeadAttOutput = np.zeros([self.q.shape[0], self.q.shape[1]])
        self.headOutputs, self.AttHinpQ,self.AttHinpK, self.AttHinpV = [], [], [], []
        for hitr in range(self.h+1):
            if hitr==0:
                valQ=np.dot(self.q,self.Wq[hitr])
                valK=np.dot(self.k,self.Wk[hitr])
                valV=np.dot(self.v,self.Wv[hitr])
                self.AttHinpQ.append(valQ)
                self.AttHinpK.append(valK)
                self.AttHinpV.append(valV)
                headAttOutput = self.AttLayer[hitr].flow(q=valQ, k=valK, v=valV)
                self.headOutputs.append(headAttOutput)
            elif hitr==self.h:
                self.ConcatHeadAttOutput = headAttOutput
            else:
                valQ=np.dot(self.q,self.Wq[hitr])
                valK=np.dot(self.k,self.Wk[hitr])
                valV=np.dot(self.v,self.Wv[hitr])
                self.AttHinpQ.append(valQ)
                self.AttHinpK.append(valK)
                self.AttHinpV.append(valV)
                val = self.AttLayer[hitr].flow(q=valQ, k=valK, v=valV)
                headAttOutput = np.concatenate((headAttOutput, val),axis=1)
                self.headOutputs.append(val)
        self.output = np.dot(self.ConcatHeadAttOutput, self.Wo)
        return self.output

    def backprop(self,derivOutput):
        dWo = np.dot(derivOutput.T, self.ConcatHeadAttOutput).T
        dConcatHeadOutput = np.dot(self.Wo, derivOutput.T).T
        dheadOutputs, dAttHinpQ, dAttHinpK, dAttHinpV = [], [], [], []
        dWq, dWk, dWv = [], [], []
        dq, dk, dv  = np.zeros([self.q.shape[0], self.q.shape[1]]), np.zeros([self.k.shape[0], self.k.shape[1]]), np.zeros([self.v.shape[0], self.v.shape[1]])
        for hitr in range(self.h):
            val = dConcatHeadOutput[:,hitr*(dmodel/self.h):(hitr+1)*(dmodel/self.h)]
            dheadOutputs.append(val)
            dvalQ, dvalK, dvalV = self.AttLayer[hitr].backprop(val)
            dAttHinpQ.append(dvalQ)
            dAttHinpK.append(dvalK)
            dAttHinpV.append(dvalV)
            dWq.append( np.dot(dvalQ.T,self.q).T )
            dWk.append( np.dot(dvalK.T,self.k).T )
            dWv.append( np.dot(dvalV.T,self.v).T )
            dq += np.dot(self.Wq[hitr], dvalQ.T).T
            dk += np.dot(self.Wk[hitr], dvalK.T).T
            dv += np.dot(self.Wv[hitr], dvalV.T).T
        # apply gradient descent
        np.clip(dWo,-5,5,out=dWo)
        self.Wo = self.Wo - learningRate*dWo
        for hitr in range(self.h):
            np.clip(dWq[hitr],-5,5,out=dWq[hitr])
            np.clip(dWk[hitr],-5,5,out=dWk[hitr])
            np.clip(dWv[hitr],-5,5,out=dWv[hitr])
            self.Wq[hitr] = self.Wq[hitr] - learningRate*dWq[hitr]
            self.Wk[hitr] = self.Wk[hitr] - learningRate*dWk[hitr]
            self.Wv[hitr] = self.Wv[hitr] - learningRate*dWv[hitr]
        return (dq+dk+dv)

class EncoderOnlyMaskedAttentionLayer:
    def flow(self,q,k,v):
        self.q = q
        self.k = k
        self.v = v
        self.a = np.dot(self.q, self.k.T)/dmodel
        for i in range(self.a.shape[0]):
            for j in range(self.a.shape[1]):
                if j>i:
                    self.a[i][j] = float('-inf')
        self.z = softmax(self.a)
        self.att = np.dot(self.z, self.v)
        return self.att

    def backprop(self, derivOutput):
        dq = np.dot(self.k.T/np.sqrt(dmodel), (np.dot(derivOutput,self.v.T)*sftderiv(self.a)).T).T
        dk = np.dot(self.q.T/np.sqrt(dmodel), (np.dot(derivOutput,self.v.T)*sftderiv(self.a)).T).T
        dv = np.dot(derivOutput.T, self.z).T
        return dq,dk,dv

class AttentionLayer:
    def flow(self, q, k, v):
        self.q = q
        self.k = k
        self.v = v
        self.z = np.dot(self.q, self.k.T)/dmodel
        self.att = np.dot(softmax(self.z), self.v)
        return self.att

class FeedFwdLayer:
    def __init__(self, hlayersize):
        self.hiddenLayerSize=hlayersize
        self.w1 = np.random.randn(dmodel, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, dmodel)
        self.b1 = np.random.randn(seq_size, self.hiddenLayerSize)
        self.b2 = np.random.randn(seq_size, dmodel)
    def flow(self,finp):
        self.inp = finp
        self.z = np.dot(self.inp, self.w1) + self.b1
        self.h = relu(self.z)
        self.y = np.dot(self.h,self.w2) + self.b2
        return self.y

    def backprop(self, derivOutput):
        dW2 = np.dot(derivOutput.T, self.h).T
        db2 = derivOutput
        dW1 = np.dot(np.dot(self.w2,derivOutput.T)*(reluderiv(self.z).T), self.inp).T
        db1 = (np.dot(self.w2,derivOutput.T)*(reluderiv(self.z).T)).T
        dinp = np.dot(self.w1, np.dot(self.w2,derivOutput.T)*(reluderiv(self.z).T)).T
        # apply gradient descent
        np.clip(dW1,-5,5,out=dW1)
        np.clip(dW2,-5,5,out=dW2)
        np.clip(db1,-5,5,out=db1)
        np.clip(db2,-5,5,out=db2)
        self.w1 = self.w1 - learningRate*dW1
        self.w2 = self.w2 - learningRate*dW2
        self.b1 = self.b1 - learningRate*db1
        self.b2 = self.b2 - learningRate*db2
        return dinp

class SimpleLinearLayer:
    def __init__(self):
        self.w = np.random.randn(dmodel, dmodel)
        self.b = np.random.randn(seq_size, dmodel)

    def flow(self,inp):
        self.inp = inp
        self.z = np.dot(self.inp, self.w) + self.b
        self.y = relu(self.z)
        self.y_curly = softmax(self.y)
        return self.y_curly

    def backprop(self,derivOutput):
        dW = np.dot((derivOutput * sftderiv(self.y) * reluderiv(self.z)).T, self.inp).T
        db = derivOutput * sftderiv(self.y) * reluderiv(self.z)
        dinp = np.dot(self.w, (sftderiv(self.y) * reluderiv(self.z) * derivOutput).T).T
        # apply gradient descent
        np.clip(dW,-5,5,out=dW)
        np.clip(db,-5,5,out=db)
        self.w = self.w - learningRate*dW
        self.b = self.b - learningRate*db
        return dinp

class Layer:
    def __init__(self):
        self.MultiAttentionLayer1 = MultiheadAttentionLayer(headsize=headSize)
        self.NormLayer1 = LayerNorm()
        self.FeedFwdLayer1 = FeedFwdLayer(hlayersize=100)
        self.NormLayer2 = LayerNorm()

    def flow(self,inp):
        self.inp = inp
        self.SubLayerOutput = self.NormLayer1.flow(self.inp + self.MultiAttentionLayer1.flow(q=self.inp,k=self.inp,v=self.inp))
        self.LayerOutput = self.NormLayer2.flow(self.SubLayerOutput + self.FeedFwdLayer1.flow(finp=self.SubLayerOutput))
        return self.LayerOutput

    def backprop(self,derivOutput):
        NL2inpDeriv = self.NormLayer2.backprop(derivOutput)
        FF1inpDeriv = self.FeedFwdLayer1.backprop(NL2inpDeriv)
        NL1inpDeriv = self.NormLayer1.backprop(FF1inpDeriv + NL2inpDeriv) # gradient flow for residual connections
        MAL1inpDeriv = self.MultiAttentionLayer1.backprop(NL1inpDeriv)
        return (MAL1inpDeriv + NL1inpDeriv) # gradient flow for residual connections

class encoder:
    def __init__(self):
        self.Layer1 = Layer()
        self.Layer2 = Layer()
        self.Layer3 = Layer()
        self.Layer4 = Layer()
        self.Layer5 = Layer()
        self.Layer6 = Layer()
        self.LinearLayer1 = SimpleLinearLayer()

    def flow(self,inp):
        self.inp = inp
        self.L1output = self.Layer1.flow(self.inp)
        self.L2output = self.Layer2.flow(self.L1output)
        self.L3output = self.Layer3.flow(self.L2output)
        self.L4output = self.Layer4.flow(self.L3output)
        self.L5output = self.Layer5.flow(self.L4output)
        self.L6output = self.Layer6.flow(self.L5output)
        self.y = self.LinearLayer1.flow(self.L6output)
        return self.y

    def backprop(self,derivOutput):
        LL1inpDeriv = self.LinearLayer1.backprop(derivOutput)
        L6inpDeriv = self.Layer6.backprop(LL1inpDeriv)
        L5inpDeriv = self.Layer5.backprop(L6inpDeriv)
        L4inpDeriv = self.Layer4.backprop(L5inpDeriv)
        L3inpDeriv = self.Layer3.backprop(L4inpDeriv)
        L2inpDeriv = self.Layer2.backprop(L3inpDeriv)
        L1inpDeriv = self.Layer1.backprop(L2inpDeriv)

################################################################################
# EXECUTION FUNCTIONS

def init():
    num_seqes = len(corpus) - seq_size
    epochs = 20
    print epochs,"Epochs |",len(corpus),"chars/words per epoch |",num_seqes,"Sequences per epoch | seq_size:",seq_size
    waste=raw_input("Press enter to start, any other key will cancel and exit the program:")
    if waste != "":
        sys.exit()
    encoder1 = encoder()
    for n in range(num_seqes*epochs):
        INP, TGT = get_seq()
        OUT = encoder1.flow(INP)
        # print OUT
        loss = KLdvgLoss(output = OUT, target = TGT, gamma = gamma) # gamma is hyperparameter (0,1)
        state = "seq no."+str(n-(int(n/num_seqes))*(num_seqes)+1)+", epoch "+str(int(n/num_seqes)+1)
        state = "Epoch no.:"+str(int(n/num_seqes)+1)+" | Seq no.:"+str(n-(int(n/num_seqes))*(num_seqes)+1)+" | Loss:"+str(loss)


        optimize(model=encoder1,output=OUT,target=TGT)


        print state
    print "\nDONE!"

def softmax(inpvec):
    # compute softmax for every row seperately
    expovec = np.exp(inpvec)
    sftvec = expovec/(np.array([np.sum(expovec,axis=1)])).T
    return sftvec

def relu(inpvec):
    return (inpvec * (inpvec > 0))

def reluderiv(inpvec):
    return (inpvec > 0)

def sftderiv(inpvec):
    expovec = np.exp(inpvec)
    sumexpovec = (np.array([np.sum(expovec,axis=1)])).T
    # print "sftderiv:expovec",expovec
    return ((expovec*sumexpovec - expovec*expovec)/(sumexpovec*sumexpovec))

def KLdvgLoss(output, target, gamma):
    # target should be 1ofK
    # output should be probabilities(via softmax), each row should add upto 1
    # both should be floats, not ints
    output = np.copy(output)
    target = np.copy(target)
    K = target.shape[1] # target is seqXemb, and as emb is 1ofK, so emb_size=no. of classes, or no. of possible chars/words
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i][j] == 1:
                target[i][j] = gamma
            else:
                target[i][j] = (1-gamma)/(K-1)
    KLloss = np.sum(target * (np.log(target) - np.log(output)), axis=1)
    seq_loss = np.sum(KLloss)/len(KLloss)
    return seq_loss

def optimize(model, output, target):
    finalDeriv = -(target/(output*seq_size))
    model.backprop(derivOutput=finalDeriv)
