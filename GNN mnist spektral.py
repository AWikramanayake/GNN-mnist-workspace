import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import tensorflow as tf

from spektral.data import Dataset, Graph
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data import BatchLoader
from spektral.datasets.mnist import MNIST
from spektral.layers import GCNConv, GlobalSumPool
from spektral.utils.sparse import sp_matrix_to_sp_tensor

# Parameters
batch_size = 64  # Batch size
epochs = 1000  # Number of training epochs
patience = 10  # Patience for early stopping
l2_reg = 5e-4  # Regularization rate for l2

# Import MNIST dataset in pixel format
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train.astype("float32")[...]/255.0, x_test.astype("float32")[...]/255.0

# Pixels with greater than 0.4x max brightness are considered 'on', all others are 'off'
x_train = np.where(0.4 < x_train, 1, 0)
x_test = np.where(0.4 < x_test, 1, 0)



# Graph Generation
# This class takes in the MNIST dataset in pixel form and converts it into a Graph format
# The details of this format can be found here: https://graphneural.network/data/#graph

class GenerateDataset(Dataset):
    def __init__(self, n_samples, data, labels, **kwargs):
        self.n_samples = n_samples
        self.data = data
        self.labels = labels
        super().__init__(**kwargs)
    
    
    def read(self):
        def make_graph(ind):
            #Flatten the 28x28 grid into a 784 length array
            #This will be immediately undone in the next step but is being done here anyway as practice for future projects
            nodes = np.ndarray.flatten(self.data[ind])
            #Isolate the indices of the bright pixels
            bright = np.delete((np.where(nodes == 1)), -1)
            
            node_coords = []
            edges = []
            
            #The divmod function returns the coordinates from the index in the 784-array
            #The quotient yields the y-coordinate and the remainder yields the x-coordinate
            for i in range(len(bright)):
                node_coords.append(divmod(bright[i],28))
            
            #This creates a matrix of the distances between the bright pixels
            DistMat = cdist(node_coords, node_coords)
            
            #This step iterates over pairs of bright pixels, and the indices of the pairs are added to 'edges' if they are within a certain distance
            #Selecting 1.5 will result edges being created between bright pixels that are within each others' neighbourhood-of-8
            #(Diagonal distance is sqrt(2) =~ 1.414)
            #The distance can be increased accordingly to include larger neighbourhoods
            for i in range(len(node_coords)):
                for j in range(i+1, len(node_coords)):
                    if DistMat[i][j] <= 1.5:
                        edges.append((bright[i],bright[j]))
                                

            # Node features
            x = np.array(nodes, dtype=np.float32)

            # Edges
            r, c = zip(*edges)
            a = sp.csr_matrix((np.ones(len(r)), (np.array(r), np.array(c))), shape=(784, 784), dtype=np.float32)
            
            # Labels
            y = self.labels[ind]
            
            # Counters and Diagnostics
            if ind == 0:
                print("started")
            elif (ind%100) == 0:
                print(ind, "graphs generated")
                
            ind +=1   
            
            return Graph(x=x, a=a, y=y)
                    
        # We must return a list of Graph objects
        return [make_graph(index) for index in range(self.n_samples)]




# Generating and shuffling the data  
# Calling this generates the train, test, and validation datasets with randomized indices      
traindata = GenerateDataset(len(x_train), x_train, y_train)
idxs = np.random.permutation(len(traindata))
split_val = int(0.85 * len(traindata))
idx_tr, idx_val = np.split(idxs, [split_val])
data_tr = traindata[idx_tr]
data_val = traindata[idx_val]

testdata = GenerateDataset(len(x_test), x_test, y_test)
idx_test = np.random.permutation(len(testdata))
data_test = testdata[idx_test]



# Data loaders
# See https://graphneural.network/loaders/ for further information
loader_tr = BatchLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = BatchLoader(data_val, batch_size=batch_size)
loader_te = BatchLoader(data_test, batch_size=batch_size)


# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(16, activation="elu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GCNConv(32, activation="elu", kernel_regularizer=l2(l2_reg))
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output
        
# Create model
model = Net()
optimizer = Adam()
loss_fn = SparseCategoricalCrossentropy()

# Training function
@tf.function
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# Evaluation function
def evaluate(loader):
    step = 0
    results = []
    for batch in loader:
        step += 1
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss = loss_fn(target, predictions)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            results = np.array(results)
            return np.average(results[:, :-1], 0, weights=results[:, -1])


# Setup training
best_val_loss = 99999
current_patience = patience
step = 0

# Training loop
results_tr = []
for batch in loader_tr:
    step += 1

    # Training step
    inputs, target = batch
    loss, acc = train_on_batch(inputs, target)
    results_tr.append((loss, acc, len(target)))

    if step == loader_tr.steps_per_epoch:
        results_va = evaluate(loader_va)
        if results_va[0] < best_val_loss:
            best_val_loss = results_va[0]
            current_patience = patience
            results_te = evaluate(loader_te)
        else:
            current_patience -= 1
            if current_patience == 0:
                print("Early stopping")
                break

        # Print results
        results_tr = np.array(results_tr)
        results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
        print(
            "Train loss: {:.4f}, acc: {:.4f} | "
            "Valid loss: {:.4f}, acc: {:.4f} | "
            "Test loss: {:.4f}, acc: {:.4f}".format(
                *results_tr, *results_va, *results_te
            )
        )

        # Reset epoch
        results_tr = []
        step = 0
