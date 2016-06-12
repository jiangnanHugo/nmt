import theano
import theano.tensor as T
import random
import numpy
import hsm

def create_data(label_count, example_size):

    the_examples_and_labels = []

    for l in range(label_count):
        base_example = []
        for i in range(example_size):
            base_example.append(random.uniform(-1.0,1.0))
        for i in range(100):
            the_examples_and_labels.append((numpy.array(corrupt(base_example)), l))

    return the_examples_and_labels

def corrupt(vector):

    new_vector = []
    for v in vector:
        new_vector.append(v + random.uniform(-0.05, 0.05))
    return new_vector

def main():

    data = create_data(100, 50)
    random.shuffle(data)
    minibatches = []
    for i in range(0,len(data),100):
        examples = []
        labels = []
        for b in range(i, i+100):
            examples.append(data[b][0])
            labels.append(data[b][1])
        minibatches.append((numpy.array(examples), numpy.array(labels)))

    print len(minibatches),len(minibatches[0]),len(minibatches[0][0]),len(minibatches[0][0][0])
    #Parallel hs-test
    tree = hsm.build_binary_tree(range(100))
    hs = hsm.HierarchicalSoftmax(tree,50,100)

    pf = hs.get_probability_function()

    print 'Minibatch probabilities before training'
    print T.exp(pf(minibatches[0][0], minibatches[0][1])[0]).eval()

    #Hierarchical softmax test
    tree = hsm.build_binary_tree(range(100))
    hs = hsm.HierarchicalSoftmax(tree,50,100)

    train_f = hs.get_training_function()

    hc = []
    for i in range(10):
        for mb in minibatches[1:]:
            hc.append(train_f(mb[0], mb[1]))
        print numpy.mean(hc), i

    pf = hs.get_probability_function()
    print 'Minibatch probabilities after training'
    print len(minibatches[0][0]),len(minibatches[0][0][0])
    print minibatches[0][1]
    print T.exp(pf(minibatches[0][0], minibatches[0][1])[0]).eval()

    print
    print 'Predictions'

    red = hs.label_tool(minibatches[0][0])
    print 'next_node','*'*40

    for item in red[0]:
        print item

    print 'labelings','*'*40
    for item in red[1]:
        print item
    print 'choice','*'*40
    for item in red[2]:
        print item
    print 'node_res_l','*'*40
    for item in red[3]:
        print item.shape
    print(numpy.sum(minibatches[0][1]==red[1][-1])*1.0/len(minibatches[0][1]))



main()