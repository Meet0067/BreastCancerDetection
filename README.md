# BreastCancerDetection Using Artificial Neural Network
Hii, There are 3 ANN Files currently uploaded

1. Cancer.py

This nueral network by default contain total 3 layers(1 Input Layer,1 Hidden Layer,1 Output Layer) .
Also By default it uses tanh activation function for Hidden Layer and Sigmoid function for Output Layer.

By executing this ANN ,you will get accuracies around 96.50% on Test set and 98.90% on Training set.
Thats it !!!

2. Cancer_Multi_Activation.py

This nueral network by default contain total 3 layers(1 Input Layer,1 Hidden Layer,1 Output Layer) As in Cancer.py file.
But here the changes comes,If you want to check model accuracies on different activation function like sigmoid , tanh , Relu 
then just Follw below stepes,
    
    =>First Change In Forward_prop() ,
            If you want Relu On Hidden Layer then simply change
            A1 = relu(Z1)
            
            If you want Sigmoid On Hidden Layer then simply change
            A1 = sigmoid(Z1)
            
            If you want Relu On Outer Layer then simply change
            A2 = relu(Z1)
            
            And so on...
       
     =>Changes In back_prop(),
            If you selected Relu on Hidden Layer then simply change
            dZ1 = relu_backward(A1,dA1)
            
            
            If you selected Sigmoid on Hidden Layer then simply change
            dZ1 = sigmoid_backward(A1,dA1)
            
            If you selected Relu on Outer Layer then simply change
            dZ2 = relu_backward(A1,dA1)
            
            And so on...

3. Cancer_Multi_Layer.py

In this ANN,you can make as many Layers as you want .For this Just change layers_dims variable at line 239
By default it is 3 layer ANN,
layers_dims = [X_train.shape[1], 23, 8, 1].

That means ,On input layer there are X_train.shape[1] number of nodes,
            On first hidden layer there are 23 nodes,
            On second hidden layer there are 8 nodes,
            On third or output layer there are 1 nodes as our problem is Binary Problem .
            
 And also you can check with different Activation function same as above file,
 but In this file you have to change in this (  L_model_forward() and L_model_backward() )  2 methods.  
 
 In This ANN default you will get 97.36% accuracies on Test set and 98.68% accuracies on Training set.
 
