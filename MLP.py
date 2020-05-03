


import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import View

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784) / 255.
    x_test = x_test.reshape(-1, 784) / 255.
    y_train = to_categorical(y_train).reshape(-1,10)
    y_test = to_categorical(y_test).reshape(-1,10)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()
features = np.concatenate((x_train,x_test),axis=0)
labels = np.concatenate((y_train,y_test),axis=0)

def pca(X_train, X_test,y_train, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(X_train,y_train)
    transform = pca.transform(X_test)
    return transform



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
n_pca = 20


X_train_pca = pca(X_train,X_train,y_train,n_pca).reshape(-1,n_pca,1)
X_test_pca = pca(X_train,X_test,y_train,n_pca).reshape(-1,n_pca,1)
y_train = y_train.reshape(-1,10,1)
y_test = y_test.reshape(-1,10,1)

def generetor_matrix(rows,cols):
    return np.int_(np.random.rand(rows,cols)*10)

def add_matrix(m1,m2):
    return m1+m2
  
def subtract_matrix(m1,m2):
    return m1-m2

def multiply_matrix(m1,m2):
    return np.dot(m1,m2)

def generetor_matrix_nn(rows,cols):
    return (np.random.rand(rows,cols)*2)-1

def hadamard(m1,m2):
    return m1*m2

def escalar_multiply(m1,x):
    return m1*x

def trasnpose(m1):
    return m1.transpose()

class NN:
    def __init__(self,i_nodes,h_nodes,o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes

        self.bias_ih = generetor_matrix_nn(self.h_nodes,1)
        self.bias_ho = generetor_matrix_nn(self.o_nodes,1)

        self.weigths_ih = generetor_matrix_nn(self.h_nodes,self.i_nodes)
        self.weigths_ho = generetor_matrix_nn(self.o_nodes,self.h_nodes)

        self.learning_rate = 0.01

    def signoid(self,x):
        return 1/(1+np.exp(-x))
  
    def dsignoid(self,x):
        return( x * (1-x))

    def train(self,input_,expected):
        #feedforward
        hidden = multiply_matrix(self.weigths_ih,input_)
        hidden = add_matrix(hidden,self.bias_ih)
        hidden = self.signoid(hidden)

        output = multiply_matrix(self.weigths_ho,hidden)
        output = add_matrix(output,self.bias_ho)
        output = self.signoid(output)

        #backpropagation

        #output to hidden
        output_error = subtract_matrix(expected,output)
        d_output = self.dsignoid(output)
        hidden_t = trasnpose(hidden)

        gradient = hadamard(d_output,output_error)
        gradient = escalar_multiply(gradient,self.learning_rate)
        #ajust Bias 0 to H
        self.bias_ho = add_matrix(self.bias_ho,gradient)

        weigths_ho_deltas = multiply_matrix(gradient,hidden_t)
        self.weigths_ho = add_matrix(self.weigths_ho,weigths_ho_deltas)


        #hidden to input
        weigths_ho_t = trasnpose(self.weigths_ho)
        hidden_error = multiply_matrix(weigths_ho_t,output_error)
        d_hidden = self.dsignoid(hidden)
        input_t = trasnpose(input_)

        gradient_h = hadamard(hidden_error,d_hidden)
        gradient_h = escalar_multiply(gradient_h,self.learning_rate)
        #ajust Bias H to I
        self.bias_ih = add_matrix(self.bias_ih,gradient_h)
        weigths_ih_deltas = multiply_matrix(gradient_h,input_t)
        self.weigths_ih = add_matrix(self.weigths_ih,weigths_ih_deltas)
        
        return self.weigths_ih, self.weigths_ho, output.argmax()
        
    def predict(self,input_,expected):
        hidden = multiply_matrix(self.weigths_ih,input_)
        hidden = add_matrix(hidden,self.bias_ih)
        hidden = self.signoid(hidden)

        output = multiply_matrix(self.weigths_ho,hidden)
        output = add_matrix(output,self.bias_ho)
        output = self.signoid(output)
        
        return output.argmax()


nn = NN(20,20,10)

y_pred = []
y_true = []
n = 30
print("Begin")
for i in range(n):
    for j in tqdm(range(len(X_train_pca[0:100]))):
        w_ih,w_ho, output = nn.train(X_train_pca[j],y_train[j])

        View.scenario(1000,
                      800,
                      20,
                      20,
                      10,
                      w_ih.reshape(400),
                      w_ho.reshape(200),
                      X_train[j].reshape(28,28),
                      output,
                      y_train[j].argmax(),
                      X_train_pca[j],
                      i,
                      j,
                      n
                     )
        

for i in (range(len(X_test_pca))):
    y_pred.append(nn.predict(X_test_pca[i],y_test[i]))
    y_true.append(y_test[i].argmax())

cm = confusion_matrix(y_pred,y_true)
acc = accuracy_score(y_pred,y_true)
recall = recall_score(y_pred,y_true,average="macro")


print(cm)
print("Accuracy:",acc)
print("Recal:",recall)

