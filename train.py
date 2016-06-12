import time
from model import *
import utils


e = 0.01
lr = 0.02
drop_rate = 0.
batch_size = 20
vocab_size=10000
embedding_dim=50
hidden_dim=200


seqs, i2w, w2i, data_xy = utils.char_sequence("data/2016.txt", batch_size)

print "compiling..."
model = Model(embedding_dim, hidden_dim,vocab_size, drop_rate)

print "training..."
start = time.time()
g_error = 9999.9999
for i in xrange(200):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X = xy[0]
        Y = xy[1]
        maskX = xy[2]
        maskY = xy[3]
        cost = model.train_model(X,maskX,Y,maskY,lr)
        print "cost:",cost
        error += cost
    in_time = time.time() - in_start

    error /= len(seqs)
    if error < g_error:
        g_error = error

    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)



