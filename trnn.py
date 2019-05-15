import numpy as np
import sys
import time
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter("error", RuntimeWarning)
################################################################################
# PARSING OF DATASET

corpus=open("seqText.txt",'r').read()[0:400000] # max 5M
# corpus=open("/content/drive/My Drive/app/seqText.txt",'r').read()[0:10000] # max 5M
chars=list(set(corpus))
emb=dict()
X,Y=[],[]
dmodel = len(chars)
for i in range(dmodel):
    emb[chars[i]]=np.zeros([dmodel])
    emb[chars[i]][i]=1

seq_size=100
beg=0
end=seq_size
def get_seq(): # generate consecutive sequences of fixed length
    global beg,end
    if end>len(corpus)-1:
        beg=0
        end=seq_size
    inp_seq, out_seq=[],[]
    for i in range(beg,end):
        inp_seq.append(emb[corpus[i]])
        out_seq.append(emb[corpus[i+1]])
    beg=end
    end+=seq_size
    return np.array(inp_seq),np.array(out_seq)

print "Sequence size :",seq_size
print "single Embedding : 1 X",len(chars)
print "single Sequence matrix input :",seq_size,"X",len(chars)
print "Total sequences : ",len(corpus)/seq_size

mychars = ['\n',' ','a','c','b','e','d','g','f','i','h','k','j','m','l','o','n','q','p','s','r','u','t','w','v','y','x','z']
# mychars = [' ','a','c','b','e','d','g','f','i','h','k','j','m','l','o','n','q','p','s','r','u','t','w','v','y','x','z']
# mychars = [' ','a','c','b','e','d','g','f','i','h','k','m','l','o','n','p','s','r','u','t','w','v','y','z']

################################################################################
# NETWORK ARCHITECTURE

EPOCH_NO = 5
learningRate = 0.001 # learning rate
gamma = 0.3 # for label smoothing using mass redistribution
headSize = 4 # dmodel/headSize must be int
guccipoint = 1.05 # has to be greater than 1 as per usage in this code, to give more weight to the a specific entry in the loss vector
HLS = 125 # embeddings expand upto this layer for each word in sequence

# for adam
warmup_steps = 4000
alpha = 0.001 # only for first update
beta1 = 0.9
beta2 = 0 # increasing as per (1-1/itr)
eps = 1e-8
EPOCH,LOSS=[],[]

print "Epochs:",EPOCH_NO

class DropoutLayer:
    def flow(self,inp):
        self.inp=inp
        self.r=np.random.binomial(n=1,p=0.2,size=[self.inp.shape[0],self.inp.shape[1]])
        self.y=self.inp*self.r
        return self.y
    def backprop(self,outderiv,t):
        dinp=outderiv*self.r
        return dinp

class LayerNorm:
    def __init__(self):
        self.gain = np.random.randn(seq_size, dmodel)
        self.bias = np.random.randn(seq_size, dmodel)
        self.gainm = np.zeros([seq_size, dmodel])
        self.gainv = np.zeros([seq_size, dmodel])
        self.biasm = np.zeros([seq_size, dmodel])
        self.biasv = np.zeros([seq_size, dmodel])

    def flow(self,x):
        self.inp = x
        self.mean = np.array([np.sum(self.inp,axis=1)/self.inp.shape[1]]).T
        self.variance = np.sqrt(np.array([np.sum((self.inp-self.mean)**2,axis=1)]).T/self.inp.shape[1])
        self.y = (self.gain/self.variance)*(self.inp-self.mean) + self.bias
        return self.y

    def backprop(self,derivOutput,t):
        dgain = derivOutput * (self.inp - self.mean)/self.variance
        dbias = derivOutput
        dinp = self.gain*(self.variance*np.ones([self.inp.shape[0],self.inp.shape[1]])*(1-1/dmodel) - (self.inp-self.mean)*2*(self.mean-self.inp)/(dmodel*dmodel*self.variance))/(self.variance*self.variance)
        # apply gradient descent

        # self.gain -= learningRate*(dgain/np.linalg.norm(dgain,'fro'))
        # self.bias -= learningRate*(dbias/np.linalg.norm(dbias,'fro'))

        alphat = alpha*(np.sqrt(1-(beta2**t))/(1-(beta1**t)))

        self.gainm = beta1*self.gainm + (1-beta1)*dgain
        self.gainv = beta2*self.gainv + (1-beta2)*dgain*dgain
        self.gain -= alphat*self.gainm/(np.sqrt(self.gainv)+eps)

        self.biasm = beta1*self.biasm + (1-beta1)*dbias
        self.biasv = beta2*self.biasv + (1-beta2)*dbias*dbias
        self.bias -= alphat*self.biasm/(np.sqrt(self.biasv)+eps)

        return dinp

class MultiheadAttentionLayer:
    def __init__(self,headsize):
        self.h = headsize
        self.Wq,self.Wk,self.Wv,self.AttLayer=[],[],[],[]
        self.Wqm, self.Wqv,self.Wkm, self.Wkv,self.Wvm, self.Wvv=[],[],[],[],[],[]
        for hitr in range(self.h):
            self.Wq.append(np.random.randn(dmodel, dmodel/self.h))
            self.Wk.append(np.random.randn(dmodel, dmodel/self.h))
            self.Wv.append(np.random.randn(dmodel, dmodel/self.h))
            self.Wqm.append(np.zeros([dmodel, dmodel/self.h]))
            self.Wqv.append(np.zeros([dmodel, dmodel/self.h]))
            self.Wkm.append(np.zeros([dmodel, dmodel/self.h]))
            self.Wkv.append(np.zeros([dmodel, dmodel/self.h]))
            self.Wvm.append(np.zeros([dmodel, dmodel/self.h]))
            self.Wvv.append(np.zeros([dmodel, dmodel/self.h]))
            self.AttLayer.append(EncoderOnlyMaskedAttentionLayer())
        self.Wo = np.random.randn(dmodel, dmodel)
        self.Wom = np.zeros([dmodel,dmodel])
        self.Wov = np.zeros([dmodel,dmodel])

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

    def backprop(self,derivOutput,t):
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

        # self.Wo -= learningRate*(dWo/np.linalg.norm(dWo,'fro'))
        # for hitr in range(self.h):
        #     self.Wq[hitr] -= learningRate*(dWq[hitr]/np.linalg.norm(dWq[hitr],'fro'))
        #     self.Wk[hitr] -= learningRate*(dWk[hitr]/np.linalg.norm(dWk[hitr],'fro'))
        #     self.Wv[hitr] -= learningRate*(dWv[hitr]/np.linalg.norm(dWv[hitr],'fro'))

        alphat = alpha*(np.sqrt(1-(beta2**t))/(1-(beta1**t)))

        self.Wom = beta1*self.Wom + (1-beta1)*dWo
        self.Wov = beta2*self.Wov + (1-beta2)*dWo*dWo
        self.Wo -= alphat*self.Wom/(np.sqrt(self.Wov)+eps)

        for hitr in range(self.h):
                self.Wqm[hitr] = beta1*self.Wqm[hitr] + (1-beta1)*dWq[hitr]
                self.Wqv[hitr] = beta2*self.Wqv[hitr] + (1-beta2)*dWq[hitr]*dWq[hitr]
                self.Wq[hitr] -= alphat*self.Wqm[hitr]/(np.sqrt(self.Wqv[hitr])+eps)

                self.Wkm[hitr] = beta1*self.Wkm[hitr] + (1-beta1)*dWk[hitr]
                self.Wkv[hitr] = beta2*self.Wkv[hitr] + (1-beta2)*dWk[hitr]*dWk[hitr]
                self.Wk[hitr] -= alphat*self.Wkm[hitr]/(np.sqrt(self.Wkv[hitr])+eps)

                self.Wvm[hitr] = beta1*self.Wvm[hitr] + (1-beta1)*dWv[hitr]
                self.Wvv[hitr] = beta2*self.Wvv[hitr] + (1-beta2)*dWv[hitr]*dWv[hitr]
                self.Wv[hitr] -= alphat*self.Wvm[hitr]/(np.sqrt(self.Wvv[hitr])+eps)
        return (dq+dk+dv)

class EncoderOnlyMaskedAttentionLayer:
    def flow(self,q,k,v):
        self.q = q
        self.k = k
        self.v = v
        self.a = np.dot(self.q, self.k.T)/np.sqrt(self.q.shape[1])
        for i in range(self.a.shape[0]):
            for j in range(self.a.shape[1]):
                if j>i:
                    self.a[i][j] = float('-inf')
        # print "MskdAtt self.a",self.a #getting large
        self.z = softmax(self.a)
        self.att = np.dot(self.z, self.v)
        return self.att

    def backprop(self, derivOutput):
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
        self.w1m = np.zeros([dmodel, self.hiddenLayerSize])
        self.w1v = np.zeros([dmodel, self.hiddenLayerSize])
        self.w2m = np.zeros([self.hiddenLayerSize, dmodel])
        self.w2v = np.zeros([self.hiddenLayerSize, dmodel])
        self.b1m = np.zeros([seq_size, self.hiddenLayerSize])
        self.b1v = np.zeros([seq_size, self.hiddenLayerSize])
        self.b2m = np.zeros([seq_size, dmodel])
        self.b2v = np.zeros([seq_size, dmodel])
    def flow(self,finp):
        self.inp = finp
        self.z = np.dot(self.inp, self.w1) + self.b1
        self.h = relu(self.z)
        self.y = np.dot(self.h,self.w2) + self.b2
        return self.y

    def backprop(self, derivOutput,t):
        dw2 = np.dot(derivOutput.T, self.h).T
        db2 = derivOutput
        dh = np.dot(self.w2,derivOutput.T).T
        dz = dh*reluDeriv(self.z)
        dw1 = np.dot(dz.T,self.inp).T
        db1 = dz
        dinp = np.dot(self.w1,dz.T).T
        # apply gradient descent

        # self.w1 -= learningRate*(dw1/np.linalg.norm(dw1,'fro'))
        # self.w2 -= learningRate*(dw2/np.linalg.norm(dw2,'fro'))
        # self.b1 -= learningRate*(db1/np.linalg.norm(db1,'fro'))
        # self.b2 -= learningRate*(db2/np.linalg.norm(db2,'fro'))

        alphat = alpha*(np.sqrt(1-(beta2**t))/(1-(beta1**t)))

        self.w1m = beta1*self.w1m + (1-beta1)*dw1
        self.w1v = beta2*self.w1v + (1-beta2)*dw1*dw1
        self.w1 -= alphat*self.w1m/(np.sqrt(self.w1v)+eps)

        self.w2m = beta1*self.w2m + (1-beta1)*dw2
        self.w2v = beta2*self.w2v + (1-beta2)*dw2*dw2
        self.w2 -= alphat*self.w2m/(np.sqrt(self.w2v)+eps)

        self.b1m = beta1*self.b1m + (1-beta1)*db1
        self.b1v = beta2*self.b1v + (1-beta2)*db1*db1
        self.b1 -= alphat*self.b1m/(np.sqrt(self.b1v)+eps)

        self.b2m = beta1*self.b2m + (1-beta1)*db2
        self.b2v = beta2*self.b2v + (1-beta2)*db2*db2
        self.b2 -= alphat*self.b2m/(np.sqrt(self.b2v)+eps)
        return dinp

class SimpleLinearLayer:
    def __init__(self):
        self.w = np.random.randn(dmodel, dmodel)
        self.b = np.random.randn(seq_size, dmodel)
        self.wm = np.zeros([dmodel,dmodel])
        self.wv = np.zeros([dmodel,dmodel])
        self.bm = np.zeros([seq_size,dmodel])
        self.bv = np.zeros([seq_size,dmodel])

    def flow(self,inp):
        self.inp = inp
        self.z = np.dot(self.inp, self.w) + self.b
        self.y = relu(self.z)
        self.y_curly = softmax(self.y)
        return self.y_curly

    def backprop(self,derivOutput,t):
        dy=derivOutput*sftderiv(self.y)
        dz=dy*reluDeriv(self.z)
        dw=np.dot(dz.T,self.inp).T
        db=dz
        dinp=np.dot(self.w,dz.T).T
        # apply gradient descent

        # self.w -= learningRate*(dw/np.linalg.norm(dw,'fro'))
        # self.b -= learningRate*(db/np.linalg.norm(db,'fro'))

        alphat = alpha*(np.sqrt(1-(beta2**t))/(1-(beta1**t)))

        self.wm = beta1*self.wm + (1-beta1)*dw
        self.wv = beta2*self.wv + (1-beta2)*dw*dw
        self.w -= alphat*self.wm/(np.sqrt(self.wv)+eps)

        self.bm = beta1*self.bm + (1-beta1)*db
        self.bv = beta2*self.bv + (1-beta2)*db*db
        self.b -= alphat*self.bm/(np.sqrt(self.bv)+eps)
        return dinp

class Layer:
    def __init__(self):
        self.MultiAttentionLayer1 = MultiheadAttentionLayer(headsize=headSize)
        self.dl1 = DropoutLayer()
        self.NormLayer1 = LayerNorm()
        self.FeedFwdLayer1 = FeedFwdLayer(hlayersize=HLS)
        self.dl2 = DropoutLayer()
        self.NormLayer2 = LayerNorm()

    def flow(self,inp):
        self.inp = inp
        self.SubLayerOutput = self.NormLayer1.flow(self.inp + self.dl1.flow(self.MultiAttentionLayer1.flow(q=self.inp,k=self.inp,v=self.inp)))
        self.LayerOutput = self.NormLayer2.flow(self.SubLayerOutput + self.dl2.flow(self.FeedFwdLayer1.flow(finp=self.SubLayerOutput)))
        return self.LayerOutput

    def backprop(self,derivOutput,t):
        NL2inpDeriv = self.NormLayer2.backprop(derivOutput,t)
        dl2inpDeriv = self.dl2.backprop(NL2inpDeriv,t)
        FF1inpDeriv = self.FeedFwdLayer1.backprop(dl2inpDeriv,t)
        NL1inpDeriv = self.NormLayer1.backprop(FF1inpDeriv + NL2inpDeriv,t) # gradient flow for residual connections
        dl1inpDeriv = self.dl1.backprop(NL1inpDeriv,t)
        MAL1inpDeriv = self.MultiAttentionLayer1.backprop(dl1inpDeriv,t)
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

    def backprop(self,derivOutput,t):
        LL1inpDeriv = self.LinearLayer1.backprop(derivOutput,t)
        L6inpDeriv = self.Layer6.backprop(LL1inpDeriv,t)
        L5inpDeriv = self.Layer5.backprop(L6inpDeriv,t)
        L4inpDeriv = self.Layer4.backprop(L5inpDeriv,t)
        L3inpDeriv = self.Layer3.backprop(L4inpDeriv,t)
        L2inpDeriv = self.Layer2.backprop(L3inpDeriv,t)
        L1inpDeriv = self.Layer1.backprop(L2inpDeriv,t)

################################################################################
# EXECUTION FUNCTIONS

encoder1 = encoder()
def init():
    global learningRate,alpha,warmup_steps, beta2
    waste=raw_input("Press enter to start, any other key will cancel and exit the program:")
    if waste != "":
        sys.exit()
    itr=0
    epoch=0
    while True:
        itr+=1
        alpha = (1/np.sqrt(500))*min(1/np.sqrt(itr),itr*(1/(np.sqrt(warmup_steps)*warmup_steps)))
        INP, TGT = get_seq()
        OUT = encoder1.flow(INP)
        loss = entropyLoss(out = OUT, tgt = TGT, gamma = gamma) # gamma is hyperparameter (0,1)
        optimize(model=encoder1,out=OUT,tgt=TGT,itr=itr)
        sys.stdout.write("\r\x1b"+str(itr))
        sys.stdout.flush()
        beta2 = (1-1.0/(itr+10))
        if itr%(len(corpus)/seq_size) == 0:
            epoch+=1
            # print "input: ",INP
            # print "target: ",TGT[0]
            # print "output: ",OUT[0]
            print ">epoch:",epoch,"| loss of last seq:",loss,"| alpha:",alpha,"| beta2:",beta2
            EPOCH.append(epoch)
            LOSS.append(loss)
        if (epoch >= EPOCH_NO):
            break
    print "\nDONE!"
    plt.plot(EPOCH,LOSS)
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS OF LAST SEQ.")
    plt.title("Dataset_size:"+str(len(corpus))+",seq_size:"+str(seq_size)+ ",no_of_sequences:"+str(len(corpus)/seq_size) +",epochs:"+str(EPOCH_NO)+",\nhiddenLayerSize:"+str(HLS)+",guccipoint:"+str(guccipoint)+",headsize:"+str(headSize))
    plt.show()

def softmax(inpvec1):
    # compute softmax for every row seperately
    inpvec = np.copy(inpvec1)
    for i in range(inpvec.shape[0]):
        inpvec[i] = inpvec[i]-max(inpvec[i])
    try:
        expovec = np.exp(inpvec)
        sftvec = expovec/(np.array([np.sum(expovec,axis=1)])).T
    except RuntimeWarning:
        print "Overflow..."
        sys.exit()
    return sftvec

def relu(inpvec):
    return (inpvec * (inpvec > 0))

def reluDeriv(inpvec):
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
    KLloss = np.sum(target * (np.log(target/output)), axis=1)
    seq_loss = np.sum(KLloss)/len(KLloss)
    return seq_loss

def entropyLoss(out,tgt,gamma):
    global guccipoint,seq_size,dmodel
    output = np.copy(out)
    target = np.copy(tgt)
    l=0
    for i in range(output.shape[0]):
        for j in range(output[i].shape[0]):
            if(target[i][j]==1):
                l += -(guccipoint)*np.log(output[i][j])
            else:
                l += -np.log(1-output[i][j])
    l/=(seq_size*dmodel)
    return l

def convert(out):
    pred1 = np.copy(out)
    predString=""
    for i in range(pred1.shape[0]):
        predString += mychars[np.argmax(pred1[i])]
    return predString

def optimize(model, out, tgt, itr):
    global guccipoint,dmodel,seq_size
    output = np.copy(out)
    target = np.copy(tgt)
    finalDeriv=np.zeros([out.shape[0],out.shape[1]])
    for i in range(target.shape[0]):
        for j in range(target[i].shape[0]):
            try:
                if(target[i][j]==1):
                    finalDeriv[i][j] = -guccipoint/(output[i][j])
                else:
                    finalDeriv[i][j] = 1/(1 - output[i][j])
            except RuntimeWarning:
                print "Overflow double-scalar"
                print output[i]
                sys.exit()
    finalDeriv /= (seq_size*dmodel)
    # K = target.shape[1] # target is seqXemb, and as emb is 1ofK, so emb_size=no. of classes, or no. of possible chars/words
    # for i in range(target.shape[0]):
    #     for j in range(target.shape[1]):
    #         if target[i][j] == 1:
    #             target[i][j] = gamma
    #         else:
    #             target[i][j] = (1-gamma)/(K-1)
    # finalDeriv = -target/(output*seq_size)
    model.backprop(derivOutput=finalDeriv,t=itr)

init()
