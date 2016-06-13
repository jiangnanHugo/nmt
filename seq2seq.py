import numpy as np
import theano
import theano.tensor as T
from gru import GRULayer
from softmax import HierarchicalSoftmaxLayer as h_softmax

# Seq2Seq Model
class Seq2Seq(object):
    # in_size: word_embedding_size
    # hidden_size: hidden_layer_size;
    # out_size: vocabulary_size;
    def __init__(self,in_size,hidden_size,source_vocab_size,target_vocab_size,p=0.5):

        self.X=T.lmatrix('batched_sequence_x')
        self.Y=T.lmatrix('batched_label_y')

        self.in_size=in_size
        self.hidden_size=hidden_size
        self.source_vocab_size=source_vocab_size
        self.target_vocab_size=target_vocab_size


        self.dropout=p
        self.is_train=T.iscalar('is_train') # for dropout
        self.maskX=T.fmatrix("maskX")
        self.maskY=T.fmatrix('maskY')

        self.wemb=theano.shared(value=np.asarray(np.random.rand(self.source_vocab_size,self.in_size),dtype=theano.config.floatX),
                                name='wemb_matrix')
        self.wemb_dec=theano.shared(value=np.asarray(np.random.rand(self.target_vocab_size,self.in_size),dtype=theano.config.floatX),
                                name='dec_wemb_matrix')

        self.build_layer()
        self.build_model()

    def build_layer(self):

        # word-embedding layers + hidden_layer
        shape=(self.in_size,self.hidden_size)
        self.encoder=GRULayer(self.wemb[self.X],self.maskX,shape,self.is_train,0)


        state_pre=self.encoder.h[-1]
        y_shifted=T.zeros_like(self.Y)
        y_shifted=T.set_subtensor(y_shifted[1:],self.Y[:-1])
        self.decoder=GRULayer(self.wemb[y_shifted],self.maskY,shape,self.is_train,self.dropout,state_pre)
        # output layer
        self.output_layer=h_softmax(
            x=self.decoder.activation,
            y=self.Y,
            maskY=self.maskY,
            shape=(self.hidden_size,self.target_vocab_size)
        )

        self.params=[self.wemb,]
        self.params+=self.encoder.params
        self.params+=self.decoder.params
        self.params+=self.output_layer.params
        self.cost=self.output_layer.activation
        self.prediction=self.output_layer.predict_labels

    def build_model(self):

        gparams=T.grad(self.cost,self.params)

        lr=T.scalar("lr")
        updates=[(p,p-lr*gp)for p,gp in zip (self.params,gparams)]



        self.train_model=theano.function(inputs=[self.X,self.maskX,self.Y,self.maskY,lr],
                                         givens={self.is_train:np.cast['int32'](1)},
                                         outputs=self.cost,
                                         updates=updates,
										 profile=True)

        self.predict_model=theano.function(inputs=[self.X,self.maskX,self.maskY],
                                           givens={self.is_train:np.cast['int32'](0)},
                                           outputs=self.prediction)



