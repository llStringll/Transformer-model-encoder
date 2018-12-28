import numpy as np
import sys
import time
import warnings

warnings.simplefilter("error", RuntimeWarning)
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

seq_size=10
beg=0
end=seq_size
def get_seq(): # generate consecutive overlapped sequqnces of fixed length
    global beg,end
    if end>len(corpus):
        beg=0
        end=seq_size
    inp_seq, out_seq=[],[]
    for i in range(beg,end):
        inp_seq.append(emb[corpus[i]])
        out_seq.append(emb[corpus[i]])
    beg=end
    end+=seq_size
    return np.array(inp_seq),np.array(out_seq)

print "Sequence size :",seq_size
print "single Embedding : 1 X",len(chars)
print "single Sequence matrix input :",seq_size,"X",len(chars)

################################################################################
# NETWORK ARCHITECTURE

learningRate = 0.001 # learning rate
gamma = 0.7 # for label smoothing using mass redistribution
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
        #print "NormLayer:",self.y
        return self.y

    def backprop(self,derivOutput):
        dgain = derivOutput * (self.inp - self.mean)/self.variance
        dbias = derivOutput
        dinp = self.gain*(self.variance*np.ones([self.inp.shape[0],self.inp.shape[1]])*(1-1/dmodel) - (self.inp-self.mean)*2*(self.mean-self.inp)/(dmodel*dmodel*self.variance))/(self.variance*self.variance)
        # apply gradient descent
        #np.clip(dgain,-5,5,out=dgain)
        #np.clip(dbias,-5,5,out=dbias)
        self.gain -= learningRate*(dgain/np.linalg.norm(dgain,'fro'))
        self.bias -= learningRate*(dbias/np.linalg.norm(dbias,'fro'))
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
        # print "self.q into MHAttn",self.q
        self.ConcatHeadAttOutput = np.zeros([self.q.shape[0], self.q.shape[1]])
        self.headOutputs, self.AttHinpQ,self.AttHinpK, self.AttHinpV = [], [], [], []
        for hitr in range(self.h+1):
            if hitr==0:
                # print "self.Wq[hitr] MHAttn",self.Wq[hitr]
                # print "self.Wk[hitr] MHAttn",self.Wk[hitr]
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
        #print "MHAtt:",self.output
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
        #np.clip(dWo,-5,5,out=dWo)
        self.Wo -= learningRate*(dWo/np.linalg.norm(dWo,'fro'))
        for hitr in range(self.h):
            #np.clip(dWq[hitr],-5,5,out=dWq[hitr])
            #np.clip(dWk[hitr],-5,5,out=dWk[hitr])
            #np.clip(dWv[hitr],-5,5,out=dWv[hitr])
            self.Wq[hitr] -= learningRate*(dWq[hitr]/np.linalg.norm(dWq[hitr],'fro'))
            self.Wk[hitr] -= learningRate*(dWk[hitr]/np.linalg.norm(dWk[hitr],'fro'))
            self.Wv[hitr] -= learningRate*(dWv[hitr]/np.linalg.norm(dWv[hitr],'fro'))
        return (dq+dk+dv)

class EncoderOnlyMaskedAttentionLayer:
    def flow(self,q,k,v):
        self.q = q
        self.k = k
        self.v = v
        # print "self.q",self.q
        # print "self.k.T",self.k.T
        self.a = np.dot(self.q, self.k.T)/np.sqrt(self.q.shape[1])
        for i in range(self.a.shape[0]):
            for j in range(self.a.shape[1]):
                if j>i:
                    self.a[i][j] = float('-inf')
        # print "MskdAtt self.a",self.a #getting large
        self.z = softmax(self.a)
        self.att = np.dot(self.z, self.v)
        #print "MaskedAtt:",self.att
        return self.att

    def backprop(self, derivOutput):
        # print "self.a MskdAtt backprop",self.a
        dz=np.dot(self.v,derivOutput.T).T
        dv=np.dot(derivOutput.T,self.z).T
        da=dz*sftderiv(self.a)
        dq=np.dot(self.k.T,da.T).T/np.sqrt(self.q.shape[1])
        dk=np.dot(da.T,self.q)/np.sqrt(self.q.shape[1])
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
        self.h = np.tanh(self.z)
        self.y = np.dot(self.h,self.w2) + self.b2
        #print "FF:",self.y
        return self.y

    def backprop(self, derivOutput):
        dw2 = np.dot(derivOutput.T, self.h).T
        db2 = derivOutput
        dh = np.dot(self.w2,derivOutput.T).T
        dz = dh*tanhDeriv(self.z)
        dw1 = np.dot(dz.T,self.inp).T
        db1 = dz
        dinp = np.dot(self.w1,dz.T).T
        # apply gradient descent
        #np.clip(dW1,-5,5,out=dW1)
        #np.clip(dW2,-5,5,out=dW2)
        #np.clip(db1,-5,5,out=db1)
        #np.clip(db2,-5,5,out=db2)
        self.w1 -= learningRate*(dw1/np.linalg.norm(dw1,'fro'))
        self.w2 -= learningRate*(dw2/np.linalg.norm(dw2,'fro'))
        self.b1 -= learningRate*(db1/np.linalg.norm(db1,'fro'))
        self.b2 -= learningRate*(db2/np.linalg.norm(db2,'fro'))
        #print "FF dW1",dW1
        return dinp

class SimpleLinearLayer:
    def __init__(self):
        self.w = np.random.randn(dmodel, dmodel)
        self.b = np.random.randn(seq_size, dmodel)

    def flow(self,inp):
        self.inp = inp
        self.z = np.dot(self.inp, self.w) + self.b
        self.y = np.tanh(self.z)
        self.y_curly = softmax(self.y)
        # print "SimpleLinear:",self.y
        return self.y_curly

    def backprop(self,derivOutput):
        dy=derivOutput*sftderiv(self.y)
        dz=dy*tanhDeriv(self.z)
        dw=np.dot(dz.T,self.inp).T
        db=dz
        dinp=np.dot(self.w,dz.T).T
        # apply gradient descent
        #np.clip(dW,-5,5,out=dW)
        #np.clip(db,-5,5,out=db)
        self.w -= learningRate*(dw/np.linalg.norm(dw,'fro'))
        self.b -= learningRate*(db/np.linalg.norm(db,'fro'))
        return dinp

class Layer:
    def __init__(self):
        self.MultiAttentionLayer1 = MultiheadAttentionLayer(headsize=headSize)
        self.NormLayer1 = LayerNorm()
        self.FeedFwdLayer1 = FeedFwdLayer(hlayersize=500)
        self.NormLayer2 = LayerNorm()

    def flow(self,inp):
        self.inp = inp
        self.SubLayerOutput = self.NormLayer1.flow(self.inp + self.MultiAttentionLayer1.flow(q=self.inp,k=self.inp,v=self.inp))
        self.LayerOutput = self.NormLayer2.flow(self.SubLayerOutput + self.FeedFwdLayer1.flow(finp=self.SubLayerOutput))
        # print "Att:",self.SubLayerOutput
        # print "EntireSingleLayer:",self.LayerOutput
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

encoder1 = encoder()
def init():
    # num_seqes = len(corpus)/seq_size
    # epochs = 500
    # print epochs,"Epochs |",len(corpus),"chars/words per epoch |",num_seqes,"Sequences per epoch | seq_size:",seq_size
    waste=raw_input("Press enter to start, any other key will cancel and exit the program:")
    if waste != "":
        sys.exit()
    itr=0
    while True:
        INP, TGT = get_seq()
        OUT = encoder1.flow(INP)
        loss = entropyLoss(out = OUT, tgt = TGT, gamma = gamma) # gamma is hyperparameter (0,1)
        # state = "seq no."+str(n-(int(n/num_seqes))*(num_seqes)+1)+", epoch "+str(int(n/num_seqes)+1)
        # state = "Epoch no.:"+str(int(n/num_seqes)+1)+" | Seq no.:"+str(n-(int(n/num_seqes))*(num_seqes)+1)+" | Loss:"+str(loss)

        optimize(model=encoder1,out=OUT,tgt=TGT)

        print convert(OUT)
        print "itr:",itr,"| loss:",loss
        itr+=1
    print "\nDONE!"

def softmax(inpvec):
    # compute softmax for every row seperately
    try:
        expovec = np.exp(inpvec)
        sftvec = expovec/(np.array([np.sum(expovec,axis=1)])).T
    except RuntimeWarning:
        print "Overflow..."
        sys.exit()
    return sftvec

def relu(inpvec):
    return (inpvec * (inpvec > 0))

def reluderiv(inpvec):
    return (inpvec > 0)

def sftderiv(inpvec):
    return (softmax(inpvec)-(softmax(inpvec)**2))

def tanhDeriv(inpvec):
    return 1-np.tanh(inpvec)**2

def KLdvgLoss(out, tgt, gamma):
    # target should be 1ofK
    # output should be probabilities(via softmax), each row should add upto 1
    # both should be floats, not ints
    output = np.copy(out)
    target = np.copy(tgt)
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

def entropyLoss(out,tgt,gamma):
    output = np.copy(out)
    target = np.copy(tgt)
    l=0
    for i in range(seq_size):
        l+= -np.log(output[i][np.argmax(target[i])])
    l/=seq_size
    return l

# mychars = ['a',' ','s','o','w']
# mychars = ['a',' ','c','b','e','l','o','n','p','s','r','t','w','y']
mychars = ['a',' ','b','l','o','n','s','w','y']

def convert(out):
    pred1 = np.copy(out)
    predString=""
    for i in range(pred1.shape[0]):
        predString += mychars[np.argmax(pred1[i])]
    return predString

def optimize(model, out, tgt):
    output = np.copy(out)
    target = np.copy(tgt)
    # finalDeriv = -(tgt/out)*seq_size
    finalDeriv=np.zeros([out.shape[0],out.shape[1]])
    for i in range(seq_size):
        finalDeriv[i][np.argmax(target[i])] = -1/(seq_size*(output[i][np.argmax(target[i])]))
    model.backprop(derivOutput=finalDeriv)

init()
