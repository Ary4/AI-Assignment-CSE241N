"""
model.py

This file implements the Logistic Regression model for classification.
"""

import numpy as np
import math
from functools import reduce

def sigmoid(x):
    """
    Sigmoid function implemented using numpy.

    @x: A numpy array of size n.

    returns: a numpy array with each element has been replaced by its sigmoid.
    """

    output = 1/(1+np.exp(-x))
    
    ''' WRITE YOUR CODE HERE (use just one line of vectorized code) '''
    
    #output = np.vectorize(lambda y: 1 / (1 + math.exp(-y)))(x)
    #print(np.shape(output))
    ''' END YOUR CODE HERE '''

    return output


class LogisticRegression(object):
    """
    A binary Logistic Regression model.
    """


    def __init__(self, add_bias = False):
        """
        Initialise the model. Here, we only initialise the weights as 'None',
        as the input size of the model will become apparent only when some
        training samples will be given to us.

        @add_bias: Whether to add bias.
        """

        # Initialise model parameters placeholders. Don't need another placeholder
        # for bias, as it can be incorporated in the weights matrix by adding
        # another feature in the input vector whose value is always 1.
        self.weights = None

        # Store the add_bias property too.
        self.add_bias = add_bias


    def loss(self, X_batch, Y_batch):
        """
        Loss function for Logistic Regression.

        Calculates the loss and gradients of weights with respect to loss
        for a batch of the logistic regression model.

        @X_batch: The input for batch, a numpy array of size NxD, where
                  N is the number of samples and D is the feature vector.
                  Assume that if 'self.add_bias' was set to True, D will
                  include a column for bias, with values always 1.
        @Y_batch: The correct labels for the batch, a numpy array of size N.

        returns: A tuple (loss, gradient).
        """

        loss, grad = 0, None

        ''' START YOUR CODE HERE '''

        # Make prediction - use the sigmoid function and self.weights (use hint.pdf in case you forget the equations involved)
        #print("x_batch{}".format(np.shape(X_batch)))
        #print("weights{}".format(np.shape(self.weights)))
        prod = np.matmul(X_batch, self.weights)
        y_pred = sigmoid(prod)
        Y_batch = np.reshape(Y_batch, (-1,1))
        '''loss_mat = -1*(np.multiply(Y_batch, np.log(y_pred)) - np.multiply((np.ones(np.shape(Y_batch)) - Y_batch)), np.log(np.ones(np.shape(y_pred)) - y_pred)
        
        loss = np.sum(loss_mat)
        loss /= np.shape(X_batch)[0]
        grad = np.dot(np.transpose(X_batch), (Y_batch - y_pred)) 
        grad /= np.shape(X_batch)[0]'''
        #print("y_batch{}".format(np.shape(Y_batch)))
        loss_mat=(-np.multiply(Y_batch,np.log(y_pred))-np.multiply(1-Y_batch,np.log(1-y_pred)))/Y_batch.size
        loss = np.sum(loss_mat)
        #print("loss{}".format(np.shape(loss_mat)))
        grad=np.dot(X_batch.T,y_pred-Y_batch)/Y_batch.size

        # calculate loss (stored in variable name loss) and gradients (stored in variable name grad)

        ''' END YOUR CODE HERE '''

        return loss, grad


    def predict(self, X_batch):
        """
        Predict the correct labels for the examples in X_batch.

        Remember: We don't need sigmoid for this. If x > 0 then sigmoid(x) > 0.5, and
        if x <= 0 then sigmoid(x) <= 0.5.

        @X_batch: The input to predict for. A numpy array of size NxD,
                  where N is the number of examples, and D is the size of input vector.
                  Since this will probably be called from outside this class,
                  we need to chech the @self.add_bias variable and adequately
                  pass the X_batch.

        returns: A vector of size N which contains 0 or 1 indicating the class label.
        """

        predict_func = np.vectorize(lambda x: 0 if x < 0 else 1)

        if self.add_bias:
            score = np.dot(np.hstack((X_batch, np.ones((X_batch.shape[0], 1)))), self.weights)
        else:
            score = np.dot(X_batch, self.weights)

        predictions = predict_func(score)
        return predictions


    def score(self, X_test, Y_test):
        """
        Score the performance of the model with the given test labels.

        @X_test: Test input numpy array of size NxD.
        @Y_test: Test correct labels numpy array of size N.

        returns: The accuracy of the model. A single float value between 0 and 1.
        """

        Y_pred = self.predict(X_test)
        Y_pred = np.reshape(Y_pred, -1,)
        accuracy = 1 - np.float(np.count_nonzero(Y_pred - Y_test)) / X_test.shape[0]

        return accuracy


    def train(self,
              X_train,
              Y_train,
              lr = 1e-3,
              reg_const = 1.0,
              num_iters = 10000,
              num_print = 100):
        """
        Learn the parameters from the given supervised samples with the given hyper parameters, using
        stochastic gradient descent.

        @X_train: Examples to learn from. It is a vector of size NxD,
                  with N being the number of samples, and D being the
                  number of features. Here, we need to check the @self.add_bias
                  flag and adequately pad the input vector with ones.
        @Y_train: Correct labels for X_train. Vector of size N.
        @lr: Learning rate for gradient descent.
        @reg_const: The regularization constant which is used to control the regularization.
        @num_iterations: Number of iterations over the training set.
        @num_print: Number of print statements during iterations.
        """

        ''' START YOUR CODE HERE '''

        # Initialise weights with normal distribution in the variable self.weights,
        # while taking in consideration the bias (so two cases according to the boolean value of add_bias).

        if self.add_bias:
            self.weights = np.hstack((np.zeros(np.shape(X_train))[1], np.ones((np.shape(X_train)[1], 1))))
        else:
            self.weights = np.zeros((np.shape(X_train)[1],1))
            

        X_train_adj = None

        # Create a new input array by adding bias column with all values one in X_train_adj array
        # while taking in consideration the bias (so two cases according to the boolean value of add_bias).
        if self.add_bias:
            X_train_adj = np.ones((np.shape(X_train)[0],(np.shape(X_train)[1]+1)))
            X_train_adj[:,:np.shape(X_train)[1]] = X_train
        else:
            X_train_adj = X_train

        # Perform iterations, use output of loss function above to get unregularized loss and gradient
        for i in range(num_iters):
            loss, grad = self.loss(X_train_adj, Y_train) 
            # Add Regularisation to the loss and grad (remember to use the reg_const being passed to the function)
            # See hint.pdf for the equations
            
            reg = reduce(lambda x,y: x+y*y, self.weights.flatten())
            reg *= reg_const/(2*np.shape(X_train)[1])
            loss = loss + reg
            grad += (reg_const/np.shape(X_train)[1])*self.weights
            # Update weights (remember to use the learning rate being passed to the function)
            # See hint.pdf for equations
            #print("grad{}".format(np.shape(grad)))
            
            self.weights -= lr*grad
            
            if i % (num_iters // num_print) == 0:
                print("Iteration %d of %d. Loss: %f" % (i, num_iters, loss))

        ''' END YOUR CODE HERE '''
