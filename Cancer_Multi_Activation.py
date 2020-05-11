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
    
    return (1/(1+np.exp(-z)))

def relu(Z): 
       
    return np.maximum(0,Z)

def layers(X,Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return n_x,n_y

def initialize(n_x,n_y,n_h):    
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    parameters = {
                "W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2     }
    
    return parameters

def forward_prop(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {
            "Z1":Z1,
            "Z2":Z2,
            "A1":A1,
            "A2":A2     }
    
    return A2,cache

def compute_cost(A2,Y,parameters):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply( (1 - Y),np.log(1 - A2))
    cost = -np.sum(logprobs)/m
    cost = float(np.squeeze(cost))
    
    return cost

def relu_backward(dA,Z):   
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.    
    dZ[Z <= 0] = 0   # When z <= 0, you should set dz to 0 as well.
 
    return dZ

def sigmoid_backward(A,dA): 
    dZ = dA * A * (1-A)
    
    return dZ

def tanh_backward(A,dA):       
    dZ = dA *(1-np.power(A,2))
    
    return dZ

def back_prop(parameters,cache,X,Y):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 =parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
      
    dZ2 = sigmoid_backward(A2,dA2)
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dA1 = np.dot(W2.T,dZ2)
    
    dZ1 = tanh_backward(A1, dA1)
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
   
    grads = {
            "dW1":dW1,
            "dW2":dW2,
            "db1":db1,
            "db2":db2,
            }
    
    return grads

def update_params(parameters ,grads,alpha):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def predict(parameters,X):
    A2,cache = forward_prop(X,parameters)
    prediction = np.round(A2)
    
    return prediction

def  model(X,y,n_h,num_iters,alpha,print_cost):
    np.random.seed(3)
    
    n_x = layers(X,y)[0]
    n_y = layers(X, y)[1]
    
    parameters = initialize(n_x,n_y,n_h)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    costs = []
    grads = {}
    
    for i in range(0,num_iters):
        A2,cache = forward_prop(X, parameters)
        cost = compute_cost(A2,y, parameters)
        grads = back_prop(parameters, cache, X,y)       
        
        if(i>20000):
            alpha1 = (20000/i)*alpha
            parameters = update_params(parameters, grads, alpha1)
        else:
            parameters = update_params(parameters, grads, alpha)
            
        if i % 100 == 0:          
            costs.append(cost)
            print("Cost after iteration "+ str(i) +"\tcost=>"+ str(cost))
        if  print_cost and i % 1000 == 0:
            
           
            if i <= 20000:
                print("Learning rate after iteration "+ str(i) +"\talpha=>"+ str(alpha))
            else:               
                print("Learning rate after iteration "+ str(i) +"\talpha=>"+ str(alpha1))    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()
    return parameters

parameters = model(X_train.T, y_train.T, n_h=20, num_iters=5000, alpha=0.0075, print_cost=True)

#Predicitng On Test Set
y_pred = predict(parameters,X_test.T)
y_pred = y_pred.reshape(y_pred.shape[1],1)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
