import numpy as np
import theano
import cPickle as pickle
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename,mode)
    return open(filename,mode)

class TextIterator:
    def __init__(self,f_source,f_target,source_dict,target_dict,
                 batch_size=100,maxlen=100,
                 n_words_source=-1, n_words_target=-1):
        self.source=fopen(f_source,'r')
        self.target=fopen(f_target,'r')
        with open(source_dict,'rb')as f:
            self.source_dict=pickle.load(f)
        with open(target_dict,'rb')as f:
            self.target_dict=pickle.load(f)

        self.batch_size=batch_size
        self.maxlen=maxlen

        self.n_words_source=n_words_source
        self.n_words_target=n_words_target

        self.source_buffer=[]
        self.target_buffer=[]
        self.end_0f_data=False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_0f_data:
            self.end_0f_data=False
            self.reset()
            raise StopIteration

        source=[]
        target=[]
        assert len(self.source_buffer)==len(self.target_buffer),'Buffer size mismatch!'

        if len(self.source_buffer)==0:
            for k_ in xrange(self.batch_size):
                ss=self.source.readline()
                tt=self.target.readline()
                if ss=="" or tt=="":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            #sort by target buffer
            tlen=np.array([len(t) for t in self.target_buffer])
            tidx=tlen.argsort()

            self.source_buffer=[self.source_buffer[i] for i in tidx]
            self.target_buffer=[self.target_buffer[i] for i in tidx]

        if len(self.source_buffer)==0 or len(self.target_buffer)==0:
            self.end_0f_data=False
            self.reset()
            raise StopIteration

        try:
            while True:
                try:
                    ss=self.source_buffer.pop()
                    tt=self.target_buffer.pop()
                except IndexError:
                    break
                ss=[self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                tt=[self.target_dict[w] if w in self.target_dict else 1 for w in tt]
                # word dict size limit
                if self.n_words_source>0:
                    ss=[w if w < self.n_words_source else 1 for w in ss]
                if self.n_words_target>0:
                    tt=[w if w < self.n_words_target else 1 for w in tt]

                # skip longer sentence
                if len(ss)>self.maxlen and len(tt)>self.maxlen:
                    continue
                source.append(ss)
                target.append(tt)

                if len(source)>=self.batch_size or len(target)>=self.batch_size:
                    break

        except IOError:
            self.end_0f_data=True

        if len(source)<=0 or len(target)<=0:
            self.end_0f_data=False
            self.reset()
            raise StopIteration

        return source,target


def prepare_data(seqs_x,seqs_y,maxlen=None,n_words_src=30000,n_words=30000):
    # x: a list of sentences
    lengths_x=[len(s) for s in seqs_x]
    lengths_y=[len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x=[]
        new_seqs_y=[]
        new_lengths_x=[]
        new_lengths_y=[]
        for l_x,s_x,l_y,s_y in zip(lengths_x,seqs_x,lengths_y,seqs_y):
            if l_x<maxlen and l_y <maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x=new_lengths_x
        seqs_x=new_seqs_x
        lengths_y=new_lengths_y
        seqs_y=new_seqs_y

        if len(lengths_x)<1 or len(lengths_y)<1:
            return None,None,None,None
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)
    maxlen_y=np.max(lengths_y)

    x=np.zeros((maxlen_x,n_samples)).astype('int64')
    y=np.zeros((maxlen_y,n_samples)).astype('int64')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')
    y_mask=np.zeros((maxlen_y,n_samples)).astype('float32')

    for idx,[s_x,s_y] in enumerate(zip(seqs_x,seqs_y)):
        x[:lengths_x[idx],idx]=s_x
        x_mask[:lengths_x[idx],idx]=1
        y[:lengths_y[idx],idx]=s_y
        y_mask[:lengths_y[idx],idx]=1
    return x,x_mask ,y,y_mask


def gene_data(f_source,f_target,source_dict,target_dict,batch_size=50):
    source=fopen(f_source,'r').read().split('\n')
    target=fopen(f_target,'r').read().split('\n')
    with open(source_dict,'rb') as f:
        w2i_x=pickle.load(f)
    with open(target_dict,'rb') as f:
        w2i_y=pickle.load(f)

    batch_x=[]
    batch_y=[]
    seqs_x=[]
    seqs_y=[]
    data_xy=[]
    for  i in xrange(len(source)):
        print '%f %%\t' %(i/len(source)),

        ss=[w2i_x[w] if w in w2i_x else 1 for w in source[i]]
        tt=[w2i_y[w] if w in w2i_y else 1 for w in target[i]]
        batch_x.append(ss)
        batch_y.append(tt)
        seqs_x.append(len(ss))
        seqs_y.append(len(tt))
        if len(batch_x)== batch_size or i == len(source)-1:
            tidx=np.array(seqs_x).argsort()
            max_x=np.max(seqs_x)
            max_y=np.max(seqs_y)
            # sort by target y
            batch_x=[batch_x[i] for i in tidx]
            batch_y=[batch_y[i] for i in tidx]
            seqs_x=[seqs_x[i] for i in tidx]
            seqs_y=[seqs_y[i] for i in tidx]

            mask_x=np.zeros((max_x,len(batch_x)),dtype=theano.config.floatX)
            mask_y=np.zeros((max_y,len(batch_y)),dtype=theano.config.floatX)
            concat_X = np.zeros((max_x, len(batch_x) ), dtype = np.int64)
            concat_Y = np.zeros((max_y, len(batch_y) ), dtype = np.int64)
            for idx,[s_x,s_y] in enumerate(zip(batch_x,batch_y)):
                concat_X[:seqs_x[idx],idx]=s_x
                concat_Y[:seqs_y[idx],idx]=s_y
                mask_x[:seqs_x[idx],idx]=1
                mask_y[:seqs_y[idx],idx]=1

            data_xy.append((concat_X, concat_Y, mask_x, mask_y))
            batch_x=[]
            batch_y=[]
            seqs_x=[]
            seqs_y=[]

    return data_xy



if __name__=='__main__':
    data_xy=gene_data(f_source='./data/europarl-v7.fr-en.en.tok.shuf',
                      f_target='./data/europarl-v7.fr-en.fr.tok.shuf',
                      source_dict='./data/europarl-v7.fr-en.en.tok.pkl',
                      target_dict='./data/europarl-v7.fr-en.fr.tok.pkl',
                      batch_size=20)
    print('Pickling....')
    with open('dataset.pkl','w') as f:
        pickle.dump(data_xy,f)

