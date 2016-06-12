import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.vector("A")

def calculate(prior,B):
    res=prior*B
    return res,prior

# Symbolic description of the result
result, updates = theano.scan(fn=calculate,
                              outputs_info=[T.ones_like(A),None],
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

print(power(range(10),2))
print(power(range(10),4))