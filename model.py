import numpy as np
import theano
import theano.tensor as T
from gru import GRULayer
from softmax import HierarchicalSoftmaxLayer as h_softmax

# next_step prediction
class Model(object):
    # in_size: word_embedding_size
    # hidden_size: hidden_layer_size;
    # out_size: vocabulary_size;
    def __init__(self,in_size,hidden_size,out_size,p=0.5):

        self.X=T.imatrix('batched_sequence_x')
        self.Y=T.imatrix('batched_label_y')
        self.lr=T.fscalar("learning_rate")

        self.in_size=in_size
        self.hidden_size=hidden_size
        self.out_size=out_size

        self.dropout=p
        self.is_train=T.iscalar('is_train') # for dropout
        self.maskX=T.matrix("maskX")
        self.maskY=T.matrix('maskY')

        self.wemb=theano.shared(value=np.asarray(np.random.rand(self.out_size,self.in_size),dtype=theano.config.floatX),
                                name='word_embedding_matrix')


        self.build_layer()
        self.build_model()

    def build_layer(self):
        # word-embedding layers + hidden_layer
        shape=(self.in_size,self.hidden_size)
        self.hidden_layer=GRULayer(self.wemb[self.X],self.maskX,shape,self.is_train,self.dropout)
        # output layer
        self.output_layer=h_softmax(
            x=self.hidden_layer.activation,
            y=self.Y,
            maskY=self.maskY,
            shape=(self.hidden_size,self.out_size)
        )

        self.params=[self.wemb,]
        self.params+=self.hidden_layer.params
        self.params+=self.output_layer.params
        self.cost=self.output_layer.activation

    def build_model(self):

        gparams=T.grad(self.cost,self.params)
        updates=[(p,p-self.lr*gp)for p,gp in zip (self.params,gparams)]
        # prediction=self.output_layer.predict_labels
        self.train_model=theano.function(inputs=[self.X,self.maskX,self.Y,self.maskY,self.lr],
                                         givens={self.is_train:np.cast['int32'](1)},
                                         outputs=self.cost,
                                         updates=updates)
        '''
        self.predict_model=theano.function(inputs=[self.X,self.maskX,self.batch_size],
                                           givens={self.is_train:np.cast['int32'](0)},
                                           outputs=prediction)
        '''


