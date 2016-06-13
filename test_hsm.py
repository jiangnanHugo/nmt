import theano
import theano.tensor as T
import random
import numpy as np
from softmax import HierarchicalSoftmaxLayer as h_softmax
import time

n_steps=20
batch_size=100
vocabulary_size=1000
hidden_size=50

def create_data(n_steps, batch_size,vocabulary_size):

    X_train=np.random.rand(n_steps,batch_size,hidden_size)
    y_train=np.random.randint(0,vocabulary_size,size=(n_steps,batch_size))
    y_mask=np.ones((n_steps,batch_size))#random.randint(0,2,size=(n_steps,batch_size))
    return np.asarray(X_train,dtype=theano.config.floatX),np.asarray(y_train,dtype=np.int),np.asarray(y_mask,dtype=np.int)


def hierarchical_softmax(X_train,y_train,y_mask):
    #Hierarchical softmax test
    x=T.tensor3('x')
    y=T.imatrix('y')
    maskY=T.imatrix('maskY')

    hs = h_softmax(x=x,y=y,shape=(hidden_size,vocabulary_size),maskY=maskY)

    train_f=theano.function(
        inputs=[x,y,maskY],
        outputs=hs.activation,
        profile=True,
        name='debug_hsm'
    )

    temp=time.time()
    activation=train_f(X_train,y_train,y_mask)
    print "activation",activation
    print("hsm time:",time.time()-temp)


if __name__=="__main__":
    X_train,y_train,y_mask = create_data(n_steps,batch_size, vocabulary_size)
    print y_train
    #normal_softmax(examples,labels)
    hierarchical_softmax(X_train,y_train,y_mask)
