#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Part-1 Data PreProcessing Phase

dataset = pd.read_csv('data.csv')
del dataset['Unnamed: 32']
X = dataset.iloc[:,2:]
Y = dataset.iloc[:,1]

#Label Encoding For M/B to 1/0  Respectively
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
Y = Y.reshape(Y.shape[0],1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part-2 Model Creation Phase

#Neural Network Initialization

def sigmoid(z):
    A =  (1/(1+np.exp(-z)))
    cache = (z)
    
    return A,cache

def tanh_(z):
    A = np.tanh(z)
    cache = (z)
    
    return A,cache

def relu(Z): 
    A = np.maximum(0,Z)  
    cache = Z 
    
    return A, cache

def relu_backward(dA, cache):    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.    
    dZ[Z <= 0] = 0  # When z <= 0, you should set dz to 0 as well. 
 
    return dZ

def sigmoid_backward(dA, cache):      
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def tanh_backward(A,dA):       
    dZ = dA *(1-np.power(A,2))
    
    return dZ

def layers(X,Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return n_x,n_y

def initialize(layer_dims):    
    np.random.seed(2)    
    parameters = { }
    L = len(layer_dims)
    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
    
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = relu(Z)        
    cache = (linear_cache,activation_cache)
    
    return A,cache

def L_model_forward(X,parameters):    
    caches = []
    A = X
    L= len(parameters) // 2
   
    for l in range(1,L):
        A_prev = A
        A, cache =linear_activation_forward(A_prev, parameters['W'+str(l)],parameters['b'+str(l)] , activation="relu")
        caches.append(cache)
        
    AL ,cache = linear_activation_forward(A, parameters['W'+str(L)],parameters['b'+str(L)] , activation="sigmoid")
    caches.append(cache)
    
    return AL,caches

def compute_cost(AL,Y):
    m = Y.shape[1]  
    logprobs = np.multiply(np.log(AL), Y) + np.multiply( (1 - Y),np.log(1 - AL))
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)   
    
    return cost

def linear_backward(dZ,cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]  
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*(np.sum(dZ,axis=1,keepdims=True))
    dA_prev = np.dot(W.T,dZ)    
   
    return dA_prev, dW, db

def linear_activation_backward(dA,cache,activation):    
    linear_cache, activation_cache = cache    
    if activation == "relu":        
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    elif activation == "sigmoid":     
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)        
  
    return dA_prev, dW, db
    
def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)         # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation  
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
   
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
   
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =linear_activation_backward(dAL,current_cache,"sigmoid")
      
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache =  caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp      

    return grads

def update_params(parameters ,grads,learning_rate):  
    L = len(parameters) //2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+ str(l+1)] -learning_rate*grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+ str(l+1)] -learning_rate*grads["db"+ str(l+1)]
        
    return parameters


def predict(X, y, parameters):    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0   
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
   
    np.random.seed(1)
    costs = []                         # keep track of cost     
    parameters = initialize(layers_dims)   
  
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.       
        AL, caches =L_model_forward(X,parameters)       
        
        # Compute cost.       
        cost = compute_cost(AL,Y)       
    
        # Backward propagation.      
        grads = L_model_backward(AL,Y,caches)    
 
        # Update parameters.      
        parameters = update_params(parameters,grads,learning_rate)
     
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

layers_dims = [X_train.shape[1], 23, 8, 1]
parameters = L_layer_model(X_train.T, y_train.T, layers_dims, num_iterations = 9000, print_cost = True)

#Predicting On X_test dataset
pred_train = predict(X_test.T,y_test.T,parameters)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_train.T)
