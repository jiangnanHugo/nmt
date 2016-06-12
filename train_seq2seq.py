import time
from seq2seq import *
from utils import TextIterator,prepare_data
import numpy as np
import cPickle as pickle


e = 0.01

drop_rate = 0.
batch_size = 20

embedding_dim=50
hidden_dim=200


dataset=['./data/europarl-v7.fr-en.en.tok.shuf',
         './data/europarl-v7.fr-en.fr.tok.shuf']
dictionary=['./data/europarl-v7.fr-en.en.tok.pkl',
            './data/europarl-v7.fr-en.fr.tok.pkl']
# source vocabulary size
source_vocab_size=10000
# target vocabulary size
target_vocab_size=10000
maxlen=100
max_epochs=100
dispFreq=1000
saveFreq=2000
lr = 0.02


print "building..."
model = Seq2Seq(embedding_dim, hidden_dim,source_vocab_size,target_vocab_size, drop_rate)

print "get dataset....."
train_dataset=TextIterator(dataset[0],dataset[1],
                           dictionary[0],dictionary[1],
                           n_words_source=source_vocab_size,n_words_target=target_vocab_size,
                           batch_size=batch_size,maxlen=maxlen)
print "Training...."
begin_again=time.time()
for eidx in xrange(max_epochs):
    uidx=0
    for x,y in train_dataset:
        uidx+=1
        x,x_mask,y,y_mask=prepare_data(x,y,maxlen,source_vocab_size,target_vocab_size)
        ud_start=time.time()
        cost = model.train_model(x,x_mask,y,y_mask,lr)
        ud=time.time()-ud_start
        if np.isnan(cost) or np.isinf(cost):
            print "Nan Detected!"

        if uidx % dispFreq==0:
            print "epoch:",eidx,'uidx',uidx,"cost:",cost

        if uidx%saveFreq==0:
            print "dumping..."
            with open('parameters_%.2f.pkl' % (time.time()-begin_again),'w')as f:
                pickle.dump(model.params,f)