import pandas
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import time

import torch
import torch.nn as nn
import torch.optim as optim

file_path = 'raw_sentences.txt' # TODO - UPDATE ME!

sentences = []
for line in open(file_path):
    words = line.split()
    sentence = [word.lower() for word in words]
    sentences.append(sentence)

vocab = set([w for s in sentences for w in s])
print(len(sentences)) # 97162
print(len(vocab)) # 250

test, valid, train = sentences[:10000], sentences[10000:20000], sentences[20000:]

words = [word for sentence in train for word in sentence]

for s in train[:10]:
    print(s)

"""
Punctiations are treated like there own words, as can be seen with
the example of the commas in the first sentence. Words with apostrophes
are treated as two words; the first half before the apostrophe, and the
second half that includes the apostrophe.
"""
vocab_counter = Counter(words)
print(vocab_counter.most_common(10))
for tup in vocab_counter.most_common(10):
    print(str(tup[0])+": "+str(round((100*tup[1])/len(words), 5))+"%")

# A list of all the words in the data set. We will assign a unique 
# identifier for each of these words.
vocab = sorted(list(set([w for s in train for w in s])))
# A mapping of index => word (string)
vocab_itos = dict(enumerate(vocab))
# A mapping of word => its index
vocab_stoi = {word:index for index, word in vocab_itos.items()}

def convert_words_to_indices(sents):
    """
    This function takes a list of sentences (list of list of words)
    and returns a new list with the same structure, but where each word
    is replaced by its index in `vocab_stoi`.

    Example:
    >>> convert_words_to_indices([['one', 'in', 'five', 'are', 'over', 'here'], ['other', 'one', 'since', 'yesterday'], ['you']])
    [[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]]
    """
    inds = []
    for sent in sents:
        temp = []
        for word in sent:
            temp.append(vocab_stoi[word])
        inds.append(temp)
    return inds

def generate_4grams(seqs):
    """
    This function takes a list of sentences (list of lists) and returns
    a new list containing the 4-grams (four consequentively occuring words)
    that appear in the sentences. Note that a unique 4-gram can appear multiple
    times, one per each time that the 4-gram appears in the data parameter `seqs`.

    Example:

    >>> generate_4grams([[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]])
    [[148, 98, 70, 23], [98, 70, 23, 154], [70, 23, 154, 89], [151, 148, 181, 246]]
    >>> generate_4grams([[1, 1, 1, 1, 1]])
    [[1, 1, 1, 1], [1, 1, 1, 1]]
    """
    grams = []
    for sent in seqs:
        for i in range(len(sent)-3):
            grams.append(sent[i:i+4])
    return grams

def process_data(sents):
    """
    This function takes a list of sentences (list of lists), and generates an
    numpy matrix with shape [N, 4] containing indices of words in 4-grams.
    """
    indices = convert_words_to_indices(sents)
    fourgrams = generate_4grams(indices)
    return np.array(fourgrams)



train4grams = process_data(train)
valid4grams = process_data(valid)
test4grams = process_data(test)

def make_onehot(indicies, total=250):
    """
    Convert indicies into one-hot vectors by
        1. Creating an identity matrix of shape [total, total]
        2. Indexing the appropriate columns of that identity matrix
    """
    I = np.eye(total)
    return I[indicies]

def softmax(x):
    """
    Compute the softmax of vector x, or row-wise for a matrix x.
    We subtract x.max(axis=0) from each row for numerical stability.
    """
    x = x.T
    exps = np.exp(x - x.max(axis=0))
    probs = exps / np.sum(exps, axis=0)
    return probs.T

def get_batch(data, range_min, range_max, onehot=True):
    """
    Convert one batch of data in the form of 4-grams into input and output
    data and return the training data (xs, ts) where:
     - `xs` is an numpy array of one-hot vectors of shape [batch_size, 3, 250]
     - `ts` is either
            - a numpy array of shape [batch_size, 250] if onehot is True,
            - a numpy array of shape [batch_size] containing indicies otherwise

    Preconditions:
     - `data` is a numpy array of shape [N, 4] produced by a call
        to `process_data`
     - range_max > range_min
    """
    xs = data[range_min:range_max, :3]
    xs = make_onehot(xs)
    ts = data[range_min:range_max, 3]
    if onehot:
        ts = make_onehot(ts).reshape(-1, 250)
    return xs, ts

def estimate_accuracy(model, data, batch_size=5000, max_N=100000):
    """
    Estimate the accuracy of the model on the data. To reduce
    computation time, use at most `max_N` elements of `data` to
    produce the estimate.
    """
    correct = 0
    N = 0
    for i in range(0, data.shape[0], batch_size):
        xs, ts = get_batch(data, i, i + batch_size, onehot=False)
        y = model(xs)
        pred = np.argmax(y, axis=1)
        correct += np.sum(ts == pred)
        N += ts.shape[0]

        if N > max_N:
            break
    return correct / N

class NumpyMLPModel(object):
    def __init__(self, num_features=250*3, num_hidden=400, num_classes=250):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.weights1 = np.zeros([num_hidden, num_features])
        self.bias1 = np.zeros([num_hidden])
        self.weights2 = np.zeros([num_classes, num_hidden])
        self.bias2 = np.zeros([num_classes])
        self.cleanup()

    def initializeParams(self):
        """
        Initialize the weights and biases of this two-layer MLP to be random.
        This random initialization is necessary to break the symmetry in the
        gradient descent update for our hidden weights and biases. If all our
        weights were initialized to the same value, then their gradients will
        all be the same!
        """
        self.weights1 = np.random.normal(0, 2/self.num_features, self.weights1.shape)
        self.bias1 = np.random.normal(0, 2/self.num_features, self.bias1.shape)
        self.weights2 = np.random.normal(0, 2/self.num_hidden, self.weights2.shape)
        self.bias2 = np.random.normal(0, 2/self.num_hidden, self.bias2.shape)

    def forward(self, inputs):
        """
        Compute the forward pass prediction for inputs.
        Note that `inputs` will be a rank-3 numpy array with shape [N, 3, 250],
        so we will need to flatten the tensor to [N, 750] first.

        For the ReLU activation, you may find the function `np.maximum` helpful
        """
        X = inputs.reshape([-1, 750])

        self.N = X.shape[0]
        self.X = X
        self.z1 =  np.matmul(self.X, (self.weights1).T) + self.bias1
        self.h = np.maximum(self.z1, np.zeros_like(self.z1))
        self.z2 = np.matmul(self.h, (self.weights2).T) + self.bias2
        self.y = softmax(self.z2)
        return self.y

    def __call__(self, inputs):
        """
        To be compatible with PyTorch API. With this code, the following two
        calls are identical:

        >>> m = TwoLayerMLP()
        >>> m.forward(inputs)

        and 

        >>> m = TwoLayerMLP()
        >>> m(inputs)
        """
        return self.forward(inputs)

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot targets.
        Note that `ts` needs to be a rank 2 numpy array with shape [N, 250].
        """
        self.z2_bar = (self.y - ts) / self.N
        self.w2_bar = np.dot(self.z2_bar.T, self.h)
        self.b2_bar = np.dot(self.z2_bar.T, np.ones(self.N))
        self.h_bar = np.matmul(self.z2_bar, self.weights2)
        self.z1_bar = self.h_bar * (self.z1 > 0)
        self.w1_bar = np.dot(self.z1_bar.T, self.X)
        self.b1_bar = np.dot(self.z1_bar.T, np.ones(self.N))

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters.
        """
        self.weights1 = self.weights1 - alpha * self.w1_bar
        self.bias1    = self.bias1    - alpha * self.b1_bar
        self.weights2 = self.weights2 - alpha * self.w2_bar
        self.bias2    = self.bias2    - alpha * self.b2_bar

    def cleanup(self):
        """
        Erase the values of the variables that we use in our computation.
        """
        self.N = None
        self.X = None
        self.z1 = None
        self.h = None
        self.z2 = None
        self.y = None
        self.z2_bar = None
        self.w2_bar = None
        self.b2_bar = None
        self.h_bar = None
        self.z1_bar = None
        self.w1_bar = None
        self.b1_bar = None



def run_gradient_descent(model,
                         train_data=train4grams,
                         validation_data=valid4grams,
                         batch_size=100,
                         learning_rate=0.1,
                         max_iters=5000):
    """
    Use gradient descent to train the numpy model on the dataset train4grams.
    """
    n = 0
    start = time.time()
    last = start
    print("")
    while n < max_iters:
        # shuffle the training data, and break early if we don't have
        # enough data to remaining in the batch
        np.random.shuffle(train_data)
        for i in range(0, train_data.shape[0], batch_size):
            if (i + batch_size) > train_data.shape[0]:
                break

            # get the input and targets of a minibatch
            xs, ts = get_batch(train_data, i, i + batch_size, onehot=True)
            
            # forward pass: compute prediction

            y = model.forward(xs)

            # backward pass: compute error 
            
            model.backward(ts)

            model.update(learning_rate)

            # increment the iteration count
            n += 1

            # compute and plot the *validation* loss and accuracy
            if (n % 100 == 0):
                train_cost = -np.sum(ts * np.log(y)) / batch_size
                train_acc = estimate_accuracy(model, train_data)
                val_acc = estimate_accuracy(model, validation_data)
                model.cleanup()
                print("Iter %d. [Val Acc %.0f%%] [Train Acc %.0f%%, Loss %f]" % (
                      n, val_acc * 100, train_acc * 100, train_cost))
                now = time.time()
                print("last iteration: "+str(round(now-last, 5))+" seconds")
                print("elapsed time: "+str(round(now-start, 5))+" seconds\n")
                last = now

            if n >= max_iters:
                return

def load_model():
    model = NumpyMLPModel()
    data = np.load("model1.npy", allow_pickle=True)
    model.weights1 = data[0]
    model.bias1 = data[1]
    model.weights2 = data[2]
    model.bias2 = data[3]
    return model
    
numpy_mlp = NumpyMLPModel()
numpy_mlp.initializeParams()
# run_gradient_descent(numpy_mlp, learning_rate=0.08, max_iters=15000)

model = load_model()

class PyTorchMLP(nn.Module):
    def __init__(self, num_hidden=400):
        super(PyTorchMLP, self).__init__()
        self.layer1 = nn.Linear(750, num_hidden)
        self.layer2 = nn.Linear(num_hidden, 250)
        self.num_hidden = num_hidden
    def forward(self, inp):
        X = inp.reshape([-1, 750])
        N = X.shape[0]
        z1 = self.layer1(X)
        h = torch.max(z1, torch.zeros_like(z1))
        z2 = self.layer2(h)
        return z2

def estimate_accuracy_torch(model, data, batch_size=5000, max_N=100000):
    """
    Estimate the accuracy of the model on the data. To reduce
    computation time, use at most `max_N` elements of `data` to
    produce the estimate.
    """
    correct = 0
    N = 0
    for i in range(0, data.shape[0], batch_size):
        # get a batch of data
        xs, ts = get_batch(data, i, i + batch_size, onehot=False)
        
        # forward pass prediction
        y = model(torch.Tensor(xs))
        y = y.detach().numpy() # convert the PyTorch tensor => numpy array
        pred = np.argmax(y, axis=1)
        correct += np.sum(pred == ts)
        N += ts.shape[0]

        if N > max_N:
            break
    return correct / N

def run_pytorch_gradient_descent(model,
                                 train_data=train4grams,
                                 validation_data=valid4grams,
                                 batch_size=100,
                                 learning_rate=0.001,
                                 weight_decay=0,
                                 max_iters=1000,
                                 checkpoint_path=None):
    """
    Train the PyTorch model on the dataset `train_data`, reporting
    the validation accuracy on `validation_data`, for `max_iters`
    iteration.

    If you want to **checkpoint** your model weights (i.e. save the
    model weights to Google Drive), then the parameter
    `checkpoint_path` should be a string path with `{}` to be replaced
    by the iteration count:

    For example, calling 

    >>> run_pytorch_gradient_descent(model, ...,
            checkpoint_path = '/content/gdrive/My Drive/CSC321/mlp/ckpt-{}.pk')

    will save the model parameters in Google Drive every 500 iterations.
    You will have to make sure that the path exists (i.e. you'll need to create
    the folder CSC321, mlp, etc...). Your Google Drive will be populated with files:

    - /content/gdrive/My Drive/CSC321/mlp/ckpt-500.pk
    - /content/gdrive/My Drive/CSC321/mlp/ckpt-1000.pk
    - ...

    To load the weights at a later time, you can run:

    >>> model.load_state_dict(torch.load('/content/gdrive/My Drive/CSC321/mlp/ckpt-500.pk'))

    This function returns the training loss, and the training/validation accuracy,
    which we can use to plot the learning curve.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    iters, losses = [], []
    iters_sub, train_accs, val_accs  = [], [] ,[]

    n = 0 # the number of iterations
    start = time.time()
    last = start
    print("")
    while True:
        for i in range(0, train_data.shape[0], batch_size):
            if (i + batch_size) > train_data.shape[0]:
                break

            # get the input and targets of a minibatch
            xs, ts = get_batch(train_data, i, i + batch_size, onehot=False)

            # convert from numpy arrays to PyTorch tensors
            xs = torch.Tensor(xs)
            ts = torch.Tensor(ts).long()

            zs = model.forward(xs)
            loss = criterion(zs, ts)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)  # compute *average* loss

            if n % 500 == 0:
                iters_sub.append(n)
                train_cost = float(loss.detach().numpy())
                train_acc = estimate_accuracy_torch(model, train_data)
                train_accs.append(train_acc)
                val_acc = estimate_accuracy_torch(model, validation_data)
                val_accs.append(val_acc)
                print("Iter %d. [Val Acc %.0f%%] [Train Acc %.0f%%, Loss %f]" % (
                      n, val_acc * 100, train_acc * 100, train_cost))
                now = time.time()
                print("last iteration: "+str(round(now-last, 5))+" seconds")
                print("elapsed time: "+str(round(now-start, 5))+" seconds\n")
                last = now
                if (checkpoint_path is not None) and n > 0:
                    torch.save(model.state_dict(), checkpoint_path.format(n))

            # increment the iteration number
            n += 1

            if n > max_iters:
                return iters, losses, iters_sub, train_accs, val_accs

def plot_learning_curve(iters, losses, iters_sub, train_accs, val_accs):
    """
    Plot the learning curve.
    """
    plt.title("Learning Curve: Loss per Iteration")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve: Accuracy per Iteration")
    plt.plot(iters_sub, train_accs, label="Train")
    plt.plot(iters_sub, val_accs, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

pytorch_mlp = PyTorchMLP()
# learning_curve_info = run_pytorch_gradient_descent(pytorch_mlp, learning_rate=0.001, max_iters=10000, checkpoint_path="ckpt-{}.pk")

# you might want to save the `learning_curve_info` somewhere, so that you can plot
# the learning curve prior to exporting your PDF file

# plot_learning_curve(*learning_curve_info)

pytorch_mlp.load_state_dict(torch.load('ckpt-10000.pk'))

def make_prediction_torch(model, sentence):
    """
    Use the model to make a prediction for the next word in the
    sentence using the last 3 words (sentence[:-3]). You may assume
    that len(sentence) >= 3 and that `model` is an instance of
    PYTorchMLP.

    This function should return the next word, represented as a string.

    Example call:
    >>> make_prediction_torch(pytorch_mlp, ['you', 'are', 'a'])
    """
    global vocab_stoi, vocab_itos

    '''
    indices = convert_words_to_indices(train)
    try:
        index1 = vocab_stoi[word1]
        index2 = vocab_stoi[word2]
        index3 = vocab_stoi[word3]
    except Exception:
        return None
    x = np.concatenate((make_onehot(index1), make_onehot(index2), make_onehot(index3)))
    y = model.forward(x)
    return vocab_itos[np.argmax(y)]
    '''
    try:
        index1 = vocab_stoi[sentence[-3]]
        index2 = vocab_stoi[sentence[-2]]
        index3 = vocab_stoi[sentence[-1]]
    except Exception:
        return None # if any given word is not in the training set, model can't predict
    x = torch.Tensor(np.concatenate((make_onehot(index1), make_onehot(index2), make_onehot(index3))))
    y = model.forward(x) # for np.argmax
    return vocab_itos[torch.argmax(y).item()]

print("") # buffer from previous print statements
print("You are a "+make_prediction_torch(pytorch_mlp, ["you", "are", "a"]))
print("few companies show "+make_prediction_torch(pytorch_mlp, ["few", "companies", "show"]))
print("There are no "+make_prediction_torch(pytorch_mlp, ["there", "are", "no"]))
print("yesterday i was "+make_prediction_torch(pytorch_mlp, ["yesterday", "i", "was"]))
print("the game had "+make_prediction_torch(pytorch_mlp, ["the", "game", "had"]))
print("yesterday the federal "+make_prediction_torch(pytorch_mlp, ["yesterday", "the", "federal"]))

print("")
print("[Test acc: "+str(round(estimate_accuracy_torch(pytorch_mlp, test4grams)*100, 3))+"%]")

class NumpyWordEmbModel(object):
    def __init__(self, vocab_size=250, emb_size=100, num_hidden=100):
        self.num_features = 3 * emb_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.num_hidden = num_hidden
        self.emb_weights = np.zeros([emb_size, vocab_size]) # no biases in this layer
        self.weights1 = np.zeros([num_hidden, emb_size * 3])
        self.bias1 = np.zeros([num_hidden])
        self.weights2 = np.zeros([vocab_size, num_hidden])
        self.bias2 = np.zeros([vocab_size])
        self.cleanup()

    def initializeParams(self):
        """
        Randomly initialize the weights and biases of this two-layer MLP.
        The randomization is necessary so that each weight is updated to
        a different value.
        """
        self.emb_weights = np.random.normal(0, 2/self.num_hidden, self.emb_weights.shape)
        self.weights1 = np.random.normal(0, 2/self.num_features, self.weights1.shape)
        self.bias1 = np.random.normal(0, 2/self.num_features, self.bias1.shape)
        self.weights2 = np.random.normal(0, 2/self.num_hidden, self.weights2.shape)
        self.bias2 = np.random.normal(0, 2/self.num_hidden, self.bias2.shape)

    def forward(self, inputs):
        """
        Compute the forward pass prediction for inputs.
        Note that `inputs` will be a rank-3 numpy array with shape [N, 3, 250].

        For numerical stability reasons, we **do not** apply the softmax
        activation in the forward function. The loss function assumes that 
        we return the logits from this function.
        """
        self.X = inputs.reshape([-1, 750])
        self.N = self.X.shape[0]
        self.pre_emb = np.zeros((self.X.shape[0], self.emb_size*3)) # I'm going to slice this in a weird way
        for i in range(3):                                                  
            self.pre_emb[:,self.emb_size*i:self.emb_size*(i+1)] = np.matmul(self.X[:,self.vocab_size*i:self.vocab_size*(i+1)], (self.emb_weights).T)
        self.emb = np.maximum(self.pre_emb, np.zeros_like(self.pre_emb)) # ReLU
        self.z1 = np.matmul(self.emb, (self.weights1).T) + self.bias1
        self.h = np.maximum(self.z1, np.zeros_like(self.z1))
        self.z2 = np.matmul(self.h, (self.weights2).T) + self.bias2
        self.y = softmax(self.z2)
        return self.y

    def __call__(self, inputs):
        return self.forward(inputs)

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot targets.
        Note that `ts` needs to be a rank 2 numpy array with shape [N, 250].

        Remember the multivariate chain rule: if a weight affects the loss
        through different paths, then the error signal from all the paths
        must be added together.
        """

        self.z2_bar = (self.y - ts) / self.N
        self.w2_bar = np.dot(self.z2_bar.T, self.h)
        self.b2_bar = np.dot(self.z2_bar.T, np.ones(self.N))
        self.h_bar = np.matmul(self.z2_bar, self.weights2)
        self.z1_bar = self.h_bar * (self.z1 > 0)
        self.w1_bar = np.dot(self.z1_bar.T, self.emb)
        self.b1_bar = np.dot(self.z1_bar.T, np.ones(self.N))
        #
        self.h0_bar = np.matmul(self.z1_bar, self.weights1)
        self.z0_bar = self.h0_bar * (self.pre_emb > 0)
        self.emb_weights_bar = np.matmul(self.w1_bar, np.matmul(self.z0_bar.T, self.X[:,:250])) + np.matmul(self.w1_bar, np.matmul(self.z0_bar.T, self.X[:,250:500])) + np.matmul(self.w1_bar, np.matmul(self.z0_bar.T, self.X[:,500:]))

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters.
        """
        self.weights1    = self.weights1       - alpha * self.w1_bar
        self.bias1       = self.bias1          - alpha * self.b1_bar
        self.weights2    = self.weights2       - alpha * self.w2_bar
        self.bias2       = self.bias2          - alpha * self.b2_bar
        self.emb_weights = self.emb_weights    - alpha * self.emb_weights_bar # indented like previous update function, looks pretty

    def cleanup(self):
        """
        Erase the values of the variables that we use in our computation.
        """
        self.N = None
        self.X = None
        self.z1 = None
        self.h = None
        self.z2 = None
        self.y = None
        self.emb = None # I just realized my naming convention is very confusing
        self.z2_bar = None
        self.w2_bar = None
        self.b2_bar = None
        self.h_bar = None
        self.z1_bar = None
        self.w1_bar = None
        self.b1_bar = None
        self.h0_bar = None
        self.z0_bar = None
        self.emb_weights_bar = None

'''
numpy_wordemb = NumpyWordEmbModel()
run_gradient_descent(numpy_wordemb, train4grams[:64], batch_size=64, max_iters=500)
'''
'''
we didn't get total convergence because we got stuck calculating the gradient for
the embedded weight, but we know that this should converge if we had the correct
gradient (we were very close to getting it, we just couldn't get the dimensions
to align)
'''
'''
run_gradient_descent(numpy_wordemb, max_iters=5000)
'''
'''
once again, no convergence for the same reason as above
'''

class PyTorchWordEmb(nn.Module):
    def __init__(self, emb_size=100, num_hidden=300, vocab_size=250):
        super(PyTorchWordEmb, self).__init__()
        self.word_emb_layer = nn.Linear(vocab_size, emb_size, bias=False)
        self.fc_layer1 = nn.Linear(emb_size * 3, num_hidden)
        self.fc_layer2 = nn.Linear(num_hidden, 250)
        self.num_hidden = num_hidden
        self.emb_size = emb_size
    def forward(self, inp):
        embeddings = torch.relu(self.word_emb_layer(inp))
        embeddings = embeddings.reshape([-1, self.emb_size * 3])
        hidden = torch.relu(self.fc_layer1(embeddings))
        return self.fc_layer2(hidden)


pytorch_wordemb= PyTorchWordEmb()

# result = run_pytorch_gradient_descent(pytorch_wordemb, max_iters=20000, checkpoint_path='ckpt2-{}.pk')

pytorch_wordemb.load_state_dict(torch.load('ckpt2-20000.pk'))
'''
# plot_learning_curve(*result)


print("") # buffer from previous print statements
print("You are a "+make_prediction_torch(pytorch_wordemb, ["you", "are", "a"]))
print("few companies show "+make_prediction_torch(pytorch_wordemb, ["few", "companies", "show"]))
print("There are no "+make_prediction_torch(pytorch_wordemb, ["there", "are", "no"]))
print("yesterday i was "+make_prediction_torch(pytorch_wordemb, ["yesterday", "i", "was"]))
print("the game had "+make_prediction_torch(pytorch_wordemb, ["the", "game", "had"]))
print("yesterday the federal "+make_prediction_torch(pytorch_wordemb, ["yesterday", "the", "federal"]))

print("")
print("[Test acc: "+str(round(estimate_accuracy_torch(pytorch_wordemb, test4grams)*100, 3))+"%]")

# Write your code here
'''


word_emb_weights = list(pytorch_wordemb.word_emb_layer.parameters())[0]
word_emb = word_emb_weights.detach().numpy().T

'''
the purpose of the word_emb_weigths matrix is to reduce the dimensionality of the
vocabulary of the input. In this particular project, we converted a 250 word vocabulary
to a 100 word vocabulary, making the word_emb_weights matrix 100x250. The matrix has
the function of transforming a higher dimensionality vocabulary into a lower dimensionality
one, and it achieves this by "grouping" words together; taking a word and transforming it
into an equivalent word in the lower dimensional space, hence each row corresponds to a
word in higher dimensional space, and each column corresponds to a word in lower dimensional
space.
'''

norms = np.linalg.norm(word_emb, axis=1)
word_emb_norm = (word_emb.T / norms).T
similarities = np.matmul(word_emb_norm, word_emb_norm.T)

# Some example distances. The first one should be larger than the second
print(similarities[vocab_stoi['any'], vocab_stoi['many']])
print(similarities[vocab_stoi['any'], vocab_stoi['government']])

maxes = []
for word in ["four", "go", "what", "should", "school", "your", "yesterday", "not"]:
    l = similarities[vocab_stoi[word], :]
    maxes.append([])
    for i in range(5):
        big = None
        for proximal in range(len(l)):
            if big == None or (l[proximal] > l[big] and proximal not in maxes[-1]):
                big = proximal
        maxes[-1].append(big)
close_words = []
for i in maxes:
    close_words.append([])
    for word in i:
        close_words[-1].append(vocab_itos[word])

examples = ["four", "go", "what", "should", "school", "your", "yesterday", "not"]

for word in range(len(examples)):
    print("'"+examples[word]+"' has the five closest words: "+str(close_words[word]))

import sklearn.manifold
tsne = sklearn.manifold.TSNE()
Y = tsne.fit_transform(word_emb)

plt.figure(figsize=(10, 10))
plt.xlim(Y[:,0].min(), Y[:, 0].max())
plt.ylim(Y[:,1].min(), Y[:, 1].max())
for i, w in enumerate(vocab):
    plt.text(Y[i, 0], Y[i, 1], w)
plt.show()

'''
What we noticed is that not only are words that have a similar letter composition placed close
together, but also words that are used in similar contexts are close together. We figure that
this could be because the similarity of sentences that each word is used in. The two clusters
we observed are the "government" cluster in the center, and the "time" cluster farther to the left.
The "government" cluster has lots of words that have an "ent" in the word such as government, percent,
department, and center, and the "time" cluster has words that relate to the concept of time such as day,
night,week, year, days, yesterday, and time.
'''
