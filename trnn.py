"""Transformer Vaswani et.al. encoder structure to get scoring for sequence, batching applied, word embeddings applied, uses KLdvgLoss"""
import numpy as np
import sys
import time
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter("error", RuntimeWarning)
#########################################################################################
# PARSING OF DATASET

print "Reading data file..."
corpus=open("seqText.txt",'r').read().replace('\n',' ').split(' ')[0:10000] # max 5M
def integer(lst):
    vec=[]
    for i in lst:
        vec.append(float(i))
    return np.array(vec)

print "Reading embeddings file..."
file='glove.6B.100d.txt'
f=open(file,'r').read().split('\n')

lookup=dict()

print "Creating lookup..."
for i in range(len(f)):
    a=f[i].split(' ')
    lookup[a[0]]=integer(a[1:])

X=[]
dmodel = len(lookup.values()[0])
seq_size=25
batch_size=25
num_batches=(len(corpus)/seq_size)/batch_size#number of batches in one epoch
Y=[]

print "Preparing custom TARGET data..."
for i in range(len(corpus)/seq_size):
    y = np.zeros([1,5])
    y[0][np.random.randint(0,5)] = 1    
    Y.append(y)
beg=0
end=seq_size

def get_seq(): # generate consecutive sequences of fixed length
    global beg,end
    if end>len(corpus):
        beg=0
        end=seq_size
    inp_seq=[]
    for i in range(beg,end):
        try:
            entry = lookup[corpus[i]]
        except KeyError:
            entry = np.random.randn(dmodel,)
        inp_seq.append(entry)
    y = Y[beg/seq_size]
    beg=end
    end+=seq_size
    return np.array(inp_seq),y

def get_batch():
    BX,BY=[],[]
    for i in range(batch_size):
        X,Y=get_seq()
        BX.append(X)
        BY.append(Y)
    return np.array(BX),np.array(BY)

print "Sequence size :",seq_size
print "single Embedding : 1 X",dmodel
print "single Sequence matrix input :",seq_size,"X",dmodel
print "Total sequences : ",len(corpus)/seq_size

#########################################################################################
# NETWORK PARAMETERS

EPOCH_NO = 5
learningRate = 0.001 # learning rate
gamma = 0.65 # for label smoothing using mass redistribution
headSize = 10 # dmodel/headSize must be int
guccipoint = 1 # has to be greater than 1 as per usage in this code, to give more weight to the a specific entry in the loss vector
HLS = 400 # embeddings expand upto this layer for each word in sequence
# for adam
warmup_steps = 100
alpha = 0.001 # only for first update
beta1 = 0.9
beta2 = 0 # increasing as per (1-1/itr)
eps = 1e-7
EPOCH,LOSS=[],[]

print "Epochs:",EPOCH_NO
print "Batch size:",batch_size

#########################################################################################
# NETWORK ARCHITECTURE

class DropoutLayer:
    def flow(self,inp):
        self.inp=inp
        self.r=np.random.binomial(n=1,p=0.3,size=[self.inp.shape[0],self.inp.shape[1],self.inp.shape[2]])
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
        self.mean = np.array(np.sum(self.inp,axis=-1)/self.inp.shape[-1])
        self.mean.resize(self.mean.shape[0],self.mean.shape[1],1)
        self.variance = np.sqrt(np.array(np.sum((self.inp-self.mean)**2,axis=-1))/self.inp.shape[-1])
        self.variance.resize(self.variance.shape[0],self.variance.shape[1],1)
        self.y = (self.gain/self.variance)*(self.inp-self.mean) + self.bias
        return self.y

    def backprop(self,derivOutput,t):
        dgain = np.sum(derivOutput * (self.inp - self.mean)/self.variance,axis=0)
        dbias = np.sum(derivOutput,axis=0)
        dvariance = np.sum(derivOutput*(self.gain*(self.inp-self.mean))*(-1/(self.variance**2)),axis=-1)
        dvariance.resize(self.variance.shape[0],self.variance.shape[1],1)
        ts = np.sum(2*(self.inp-self.mean),axis=-1)
        ts.resize(ts.shape[0],ts.shape[1],1)
        dmean = np.sum(derivOutput*( (self.gain/self.variance)*(-1) + dvariance*(ts/(2*self.variance*self.inp.shape[-1])) ),axis=-1)
        dmean.resize(self.mean.shape[0],self.mean.shape[1],1)
        dinp = derivOutput*( self.gain/self.variance + 1/self.inp.shape[-1] + 2*(self.inp-self.mean)/(2*self.variance*self.inp.shape[-1]) )
        # apply gradient descent

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
        self.ConcatHeadAttOutput = np.zeros([self.q.shape[0], self.q.shape[1],self.q.shape[2]])
        self.headOutputs, self.AttHinpQ,self.AttHinpK, self.AttHinpV = [], [], [], []
        for hitr in range(self.h+1):
            if hitr==0:
                valQ=matmul(self.q,self.Wq[hitr])
                valK=matmul(self.k,self.Wk[hitr])
                valV=matmul(self.v,self.Wv[hitr])
                self.AttHinpQ.append(valQ)
                self.AttHinpK.append(valK)
                self.AttHinpV.append(valV)
                headAttOutput = self.AttLayer[hitr].flow(q=valQ, k=valK, v=valV)
                self.headOutputs.append(headAttOutput)
            elif hitr==self.h:
                self.ConcatHeadAttOutput = headAttOutput
            else:
                valQ=matmul(self.q,self.Wq[hitr])
                valK=matmul(self.k,self.Wk[hitr])
                valV=matmul(self.v,self.Wv[hitr])
                self.AttHinpQ.append(valQ)
                self.AttHinpK.append(valK)
                self.AttHinpV.append(valV)
                val = self.AttLayer[hitr].flow(q=valQ, k=valK, v=valV)
                headAttOutput = np.concatenate((headAttOutput, val),axis=-1)
                self.headOutputs.append(val)
        self.output = matmul(self.ConcatHeadAttOutput, self.Wo)
        return self.output

    def backprop(self,derivOutput,t):
        dWo = np.sum(matmul(derivOutput, self.ConcatHeadAttOutput,t='a'),axis=0).T
        dConcatHeadOutput = matmul(derivOutput,self.Wo,t='b')
        dheadOutputs, dAttHinpQ, dAttHinpK, dAttHinpV = [], [], [], []
        dWq, dWk, dWv = [], [], []
        dq, dk, dv  = np.zeros([self.q.shape[0], self.q.shape[1],self.q.shape[2]]), np.zeros([self.k.shape[0], self.k.shape[1],self.k.shape[2]]), np.zeros([self.v.shape[0], self.v.shape[1],self.v.shape[2]])
        for hitr in range(self.h):
            val = dConcatHeadOutput[:,:,hitr*(dmodel/self.h):(hitr+1)*(dmodel/self.h)]
            dheadOutputs.append(val)
            dvalQ, dvalK, dvalV = self.AttLayer[hitr].backprop(val)
            dAttHinpQ.append(dvalQ)
            dAttHinpK.append(dvalK)
            dAttHinpV.append(dvalV)
            dWq.append( np.sum(matmul(self.q,dvalQ,t='a'),axis=0) )
            dWk.append( np.sum(matmul(self.k,dvalK,t='a'),axis=0) )
            dWv.append( np.sum(matmul(self.v,dvalV,t='a'),axis=0) )
            dq += matmul(dvalQ,self.Wq[hitr],t='b')
            dk += matmul(dvalK,self.Wk[hitr],t='b')
            dv += matmul(dvalV,self.Wv[hitr],t='b')
        # apply gradient descent

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
        self.a = matmul(self.q, self.k, t='b')/np.sqrt(self.q.shape[-1])
        self.z = softmax(self.a)
        self.att = matmul(self.z, self.v)
        return self.att

    def backprop(self, derivOutput):
        dz=matmul(derivOutput,self.v,t='b')
        dv=matmul(self.z,derivOutput,t='a')
        da=np.zeros([self.a.shape[0],self.a.shape[1],self.a.shape[2]])
        dsft=sftderiv(self.a)
        da = matmul(dsft,dz)
        dq=matmul(da,self.k)/np.sqrt(self.q.shape[-1])
        dk=matmul(da,self.q,t='a')/np.sqrt(self.q.shape[-1])
        return dq,dk,dv

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
        self.z = matmul(self.inp, self.w1) + self.b1
        self.h = relu(self.z)
        self.y = matmul(self.h,self.w2) + self.b2
        return self.y

    def backprop(self, derivOutput,t):
        dw2 = np.sum(matmul(derivOutput, self.h,t='a'),axis=0).T
        db2 = np.sum(derivOutput,axis=0)
        dh = matmul(derivOutput,self.w2,t='b')
        dz = dh*reluDeriv(self.z)
        dw1 = np.sum(matmul(dz,self.inp,t='a'),axis=0).T
        db1 = np.sum(dz,axis=0)
        dinp = matmul(dz,self.w1,t='b')
        # apply gradient descent

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
        self.w1 = np.random.randn(1,seq_size)
        self.b1 = np.zeros([1,dmodel])
        self.w2 = np.random.randn(dmodel,dmodel/4)
        self.b2 = np.zeros([1,dmodel/4])
        self.w3 = np.random.randn(dmodel/4,dmodel/4)
        self.b3 = np.zeros([1,dmodel/4])
        self.w4 = np.random.randn(dmodel/4,dmodel/4)
        self.b4 = np.zeros([1,dmodel/4])        
        self.w5 = np.random.randn(dmodel/4,5)
        self.b5 = np.zeros([1,5])
        self.DL1 = DropoutLayer()
        self.DL2 = DropoutLayer()
        self.DL3 = DropoutLayer()
        self.w1m = np.zeros([1,seq_size])
        self.w1v = np.zeros([1,seq_size])
        self.b1m = np.zeros([1,dmodel])
        self.b1v = np.zeros([1,dmodel])
        self.w2m = np.zeros([dmodel,dmodel/4])
        self.w2v = np.zeros([dmodel,dmodel/4])
        self.b2m = np.zeros([1,dmodel/4])
        self.b2v = np.zeros([1,dmodel/4])
        self.w3m = np.zeros([dmodel/4,dmodel/4])
        self.w3v = np.zeros([dmodel/4,dmodel/4])
        self.b3m = np.zeros([1,dmodel/4])
        self.b3v = np.zeros([1,dmodel/4])
        self.w4m = np.zeros([dmodel/4,dmodel/4])
        self.w4v = np.zeros([dmodel/4,dmodel/4])
        self.b4m = np.zeros([1,dmodel/4])
        self.b4v = np.zeros([1,dmodel/4])        
        self.w5m = np.zeros([dmodel/4,5])
        self.w5v = np.zeros([dmodel/4,5])
        self.b5m = np.zeros([1,5])
        self.b5v = np.zeros([1,5])        

    def flow(self,inp):
        self.inp = inp # sXe
        self.z1 = matmul(self.w1,self.inp) + self.b1
        self.a1 = softmax(self.z1)
        self.z2 = matmul(self.a1,self.w2) + self.b2
        self.a2 = relu(self.z2)
        self.a2d = self.DL1.flow(self.a2)
        self.z3 = matmul(self.a2d,self.w3) + self.b3
        self.a3 = relu(self.z3)
        self.a3d = self.DL2.flow(self.a3)
        self.z4 = matmul(self.a3d,self.w4) + self.b4
        self.a4 = relu(self.z4)
        self.a4d = self.DL3.flow(self.a4)
        self.logits = matmul(self.a4d,self.w5) + self.b5
        self.y = softmax(self.logits)
        return self.y

    def backprop(self,derivOutput,t):
        dsft = sftderiv(self.logits)
        dlogits = np.zeros([self.logits.shape[0],self.logits.shape[1],self.logits.shape[2]])
        dlogits = matmul(dsft,derivOutput)
        db5 = np.sum(dlogits,axis=0)
        dw5 = np.sum(matmul(self.a4d,dlogits,t='a'),axis=0)
        da4d = matmul(dlogits,self.w5,t='b')
        da4 = self.DL3.backprop(da4d,t)
        dz4 = da4*reluDeriv(self.z4)
        db4 = np.sum(dz4,axis=0)
        dw4 = np.sum(matmul(self.a3d,dz4,t='a'),axis=0)
        da3d = matmul(dz4,self.w4,t='b')
        da3 = self.DL2.backprop(da3d,t)
        dz3 = da3*reluDeriv(self.z3)
        db3 = np.sum(dz3,axis=0)
        dw3 = np.sum(matmul(self.a2d,dz3,t='a'),axis=0)
        da2d = matmul(dz3,self.w3,t='b')
        da2 = self.DL2.backprop(da2d,t)
        dz2 = da2*reluDeriv(self.z2)
        db2 = np.sum(dz2,axis=0)
        dw2 = np.sum(matmul(self.a1,dz2,t='a'),axis=0)
        da1 = matmul(dz2,self.w2,t='b')
        dz1 = matmul(sftderiv(self.z1),da1)
        db1 = np.sum(dz1,axis=0)
        dw1 = np.sum(matmul(dz1,self.inp,t='b'),axis=0)
        dinp = matmul(self.w1,dz1,t='a')

        alphat = alpha*(np.sqrt(1-(beta2**t))/(1-(beta1**t)))

        self.w1m = beta1*self.w1m + (1-beta1)*dw1
        self.w1v = beta2*self.w1v + (1-beta2)*dw1*dw1
        self.w1 -= alphat*self.w1m/(np.sqrt(self.w1v)+eps)

        self.w2m = beta1*self.w2m + (1-beta1)*dw2
        self.w2v = beta2*self.w2v + (1-beta2)*dw2*dw2
        self.w2 -= alphat*self.w2m/(np.sqrt(self.w2v)+eps)

        self.w3m = beta1*self.w3m + (1-beta1)*dw3
        self.w3v = beta2*self.w3v + (1-beta2)*dw3*dw3
        self.w3 -= alphat*self.w3m/(np.sqrt(self.w3v)+eps)

        self.w4m = beta1*self.w4m + (1-beta1)*dw4
        self.w4v = beta2*self.w4v + (1-beta2)*dw4*dw4
        self.w4 -= alphat*self.w4m/(np.sqrt(self.w4v)+eps)

        self.w5m = beta1*self.w5m + (1-beta1)*dw5
        self.w5v = beta2*self.w5v + (1-beta2)*dw5*dw5
        self.w5 -= alphat*self.w5m/(np.sqrt(self.w5v)+eps)                

        self.b1m = beta1*self.b1m + (1-beta1)*db1
        self.b1v = beta2*self.b1v + (1-beta2)*db1*db1
        self.b1 -= alphat*self.b1m/(np.sqrt(self.b1v)+eps)

        self.b2m = beta1*self.b2m + (1-beta1)*db2
        self.b2v = beta2*self.b2v + (1-beta2)*db2*db2
        self.b2 -= alphat*self.b2m/(np.sqrt(self.b2v)+eps)

        self.b3m = beta1*self.b3m + (1-beta1)*db3
        self.b3v = beta2*self.b3v + (1-beta2)*db3*db3
        self.b3 -= alphat*self.b3m/(np.sqrt(self.b3v)+eps)

        self.b4m = beta1*self.b4m + (1-beta1)*db4
        self.b4v = beta2*self.b4v + (1-beta2)*db4*db4
        self.b4 -= alphat*self.b4m/(np.sqrt(self.b4v)+eps)

        self.b5m = beta1*self.b5m + (1-beta1)*db5
        self.b5v = beta2*self.b5v + (1-beta2)*db5*db5
        self.b5 -= alphat*self.b5m/(np.sqrt(self.b5v)+eps)                

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
    global learningRate,alpha,warmup_steps,beta2,eps
    waste=raw_input("Press enter to start, any other key will cancel and exit the program:")
    if waste != "":
        sys.exit()
    start_time = time.time()
    itr=0
    epoch=0
    loss = 0
    while True:
        itr+=1
        alpha = (1/np.sqrt(dmodel))*min(1/np.sqrt(itr),itr*(1/(np.sqrt(warmup_steps)*warmup_steps)))
        INP, TGT = get_batch()
        OUT = encoder1.flow(pos_enc(INP))
        loss += LossandOptimize(model=encoder1, out=OUT, tgt=TGT, itr=itr)
        frac=itr*1.0/(EPOCH_NO*num_batches)
        mystr = '\rProgress:'+'[{:>8.3%}]'.format(frac)
        sys.stdout.write(mystr)
        sys.stdout.flush()
        beta2 = (1-1.0/(itr+10))
        if itr%num_batches == 0:
            epoch+=1
            accLoss = accuracy(model=encoder1)
            print ">epoch:",epoch,"| loss:",accLoss,"| alpha:",alpha,"| beta2:",beta2
            EPOCH.append(epoch)
            LOSS.append(accLoss)
            loss = 0
        if (epoch >= EPOCH_NO):
            break
    print "\nDONE!, elapsed time:",(time.time()-start_time)
    plt.plot(EPOCH,LOSS)
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.show()

def softmax(inpvec1):
    # compute softmax for every row seperately
    inpvec = np.copy(inpvec1)
    for i in range(inpvec.shape[0]):
        for j in range(inpvec.shape[1]):
            inpvec[i][j] = inpvec[i][j]-max(inpvec[i][j])
    try:
        expovec = np.exp(inpvec)
        s=np.sum(expovec,axis=-1)
        s.resize(expovec.shape[0],expovec.shape[1],1)
        sftvec = expovec/s
    except RuntimeWarning:
        print "Overflow..."
        sys.exit()
    return sftvec

def relu(inpvec):
    return (inpvec * (inpvec > 0))

def reluDeriv(inpvec):
    return (inpvec > 0)

def sftderiv(inpvec):
    deriv = np.zeros([inpvec.shape[0],inpvec.shape[1],inpvec.shape[2],inpvec.shape[2]])
    p = softmax(inpvec)
    for b in range(inpvec.shape[0]):
        for s in range(inpvec.shape[1]):
                for i in range(inpvec.shape[2]):
                    for j in range(inpvec.shape[2]):
                        if i==j:
                            deriv[b][s][i][j] = p[b][s][i]*(1-p[b][s][j])
                        else:
                            deriv[b][s][i][j] = -p[b][s][i]*p[b][s][j]

    return deriv

# def LossandOptimize(model,out,tgt,itr):
#     global guccipoint,seq_size,dmodel,batch_size
#     output = np.copy(out)
#     target = np.copy(tgt)
#     l=0
#     finalDeriv=np.zeros([output.shape[0],output.shape[1],output.shape[2]])
#     for b in range(output.shape[0]):
#         for s in range(output.shape[1]):
#             lt = -np.log(output[b][s][np.argmax(target[b][s])])
#             l += lt
#             finalDeriv[b][s][np.argmax(target[b][s])] = -1/(output[b][s][np.argmax(target[b][s])])            
#     model.backprop(derivOutput=finalDeriv,t=itr)            
#     return l

def LossandOptimize(model,out,tgt,itr):
    """USES KLdvgLoss"""
    global guccipoint,gamma,seq_size,dmodel,batch_size
    output = np.copy(out)
    target = np.copy(tgt)
    for _b in range(batch_size):
        for _s in range(target.shape[1]):
            for _e in range(target.shape[-1]):
                if target[_b][_s][_e]==1:
                    target[_b][_s][_e] = gamma
                else:
                    target[_b][_s][_e] = (1-gamma)/(target.shape[-1]-1)
    finalDeriv=np.zeros([output.shape[0],output.shape[1],output.shape[2]])
    l = np.sum(-target*np.log(output/target))/batch_size
    finalDeriv = -(target/output)/batch_size
    model.backprop(derivOutput=finalDeriv,t=itr)            
    return l

def convert(out):
    pred1 = np.copy(out)
    predString=""
    for i in range(pred1.shape[0]):
        predString += mychars[np.argmax(pred1[i])]
    return predString

def accuracy(model):
    global gamma,seq_size
    abeg = np.random.randint(0,(len(corpus)/seq_size))*seq_size
    aend = abeg+seq_size
    ainp_seq=[]
    for i in range(abeg,aend):
        try:
            aentry = lookup[corpus[i]]
        except KeyError:
            aentry = np.random.randn(dmodel,)
        ainp_seq.append(aentry)
    ay = Y[abeg/seq_size]
    ay = np.array([ay])
    ainp_seq = np.array(ainp_seq)
    ainp_seq = np.array([ainp_seq])
    aout = model.flow(pos_enc(ainp_seq))

    output = np.copy(aout)
    target = np.copy(ay)
    for _b in range(1):
        for _s in range(target.shape[1]):
            for _e in range(target.shape[-1]):
                if target[_b][_s][_e]==1:
                    target[_b][_s][_e] = gamma
                else:
                    target[_b][_s][_e] = (1-gamma)/(target.shape[-1]-1)
    l = np.sum(-target*np.log(output/target))/1
    print "\nAccuracy metric>>>> output:",aout,", target:",ay
    return l    


def matmul(a,b,t=None):
    """a is input, b is weight"""
    out=[]
    if len(a.shape)==len(b.shape):
        for i in range(a.shape[0]):
            if t=='b':
                out.append(np.dot(a[i],b[i].T))
            elif t=='a':
                out.append(np.dot(a[i].T,b[i]))
            else:
                out.append(np.dot(a[i],b[i]))
    elif len(a.shape)>len(b.shape) and len(a.shape)==4:
        B=[]
        for _b in range(a.shape[0]):
            S=[]
            for _s in range(a.shape[1]):
                E = np.dot(a[_b][_s],b[_b][_s])
                S.append(E)
            S=np.array(S)
            B.append(S)
        out=B
    elif len(a.shape)>len(b.shape):
        for i in range(a.shape[0]):
            if t=='a':
                out.append(np.dot(a[i].T,b))
            elif t=='b':
                out.append(np.dot(a[i],b.T))
            else:
                out.append(np.dot(a[i],b))
    elif len(b.shape)>len(a.shape):
        for i in range(b.shape[0]):
            if t=='a':
                out.append(np.dot(a.T,b[i]))
            elif t=='b':
                out.append(np.dot(a,b[i].T))
            else:
                out.append(np.dot(a,b[i]))
    out = np.array(out)
    return out

def pos_enc(inp):
    pe = np.zeros([seq_size, dmodel])
    position = np.arange(0, seq_size)
    position.resize(seq_size,1)
    div_term = np.exp(np.arange(0, dmodel, 2) * -(np.log(10000.0) / dmodel))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return inp+pe

init()
