import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# JONAS NUNES
# https://github.com/joneco02
# Batch Gradient Descent

# Reading dataset. This dataset relates a city population with the profit of a company
data = pd.read_csv("ex2data1.txt", header=None)
X=data.iloc[:, 0].values
y=data.iloc[:, 1].values

#avoiding a tensor rank-1 (vector)
X=X.reshape(X.shape[0],1)
y=y.reshape(y.shape[0],1)

# its possible to see that booth data are column vectors
print(X.shape,y.shape) 

# batch gradient descent cost function).
def  costfunction(theta,X,y):
    #   theta is the weight vectors including bias
    #   y is the output
    
    m = len(y)
    # output prediction
    y_pred = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(y_pred-y))
    return cost

def gradient_descente(X,y,theta,lr=0.01,nit=100):
    #   theta is the weight vector including bias
    #   X is the Column Vector X
    #   y is the Column Vector Y
    #   lr is the learning rate 
    #   nit is the number of iterations

    number_of_elements = len(y)
    dim=X.shape[1]
    cost_hist = np.zeros(nit)
    theta_hist = np.zeros((nit,dim+1))
    X_padding = np.c_[np.ones((len(X),1)),X]
    
    for it in range(nit):
        # output prediction
        y_pred = np.dot(X_padding,theta)
        # gradient calculation
	gradient = ( X_padding.T.dot( (y_pred - y) ))
        # new theta value for this iteration
	theta = theta - (1/m) * lr * gradient
        theta_hist[it,:] =theta.T
        cost_hist[it]  = costfunction(theta,X_padding,y)
    return theta, cost_hist, theta_hist
	
# as an example for comparison, we can try to reduce the costfunction using the gradient descent with all weights set to ZERO .
X_padding = np.c_[np.ones((len(X),1)),X]
theta = np.zeros((X_padding.shape[1],1))
cost=costfunction(theta,X_padding,y)
print('costfunction :  {:0.3f}'.format(cost))

# now we will use one dataset and minimize the costfunction using gradient descent to plot the gradient lined) Faça o gráfico da reta de regressão para o conjunto de dados fornecido.
# we will the learning rate hyper-parameter.

# LR is used to control the rate at which an algorithm updates the parameter estimates or learns the values of the parameters.   
lr =0.001
# number of iterations.
n_iter = 5000

theta = np.random.randn(X.shape[1]+1,1)
theta,cost_history,theta_history = gradient_descente(X,y,theta,lr,n_iter)

print('Theta0:        {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('costfunction:  {:0.3f}'.format(cost_history[-1]))

y_pred= X_padding.dot(theta)
plot.plot(X,y_pred,'r-')
plot.plot(X,y,'b.')
plot.xlabel("Population Number", fontsize=18)
plot.ylabel("Enterprise Profit", fontsize=18)
plot.show()

# We can try to predict the output for 2 different data sets
# X1 = [1,3.5]
X1_padding=np.array([1 , 3.5]).reshape(1,2)
# X2 = [1,7]
X2_padding=np.array([1 , 7]).reshape(1,2)
# prediction for the dataset X1

y1_pred=X1_padding.dot(theta)
# prediction for the dataset X2
y2_pred=X2_padding.dot(theta)

# plotting the training dataset in blue and the y1 e y2 predictions in red
plot.plot(X1_padding[0][1],y1_pred,'ro')
plot.plot(X2_padding[0][1],y2_pred,'ro')
plot.plot(X,y,'b.')
plot.xlabel("Population Number", fontsize=18)
plot.ylabel("Enterprise Profit", fontsize=18)
plot.show()

# prediction for X1
print(X2_padding[0][1],y2_pred[0][0])

# prediction for X2
print(X1_padding[0][1],y1_pred[0][0])

# printing some points of the costfunction for different values of epochs values.
print(cost_history[200])
print(cost_history[300])
print(cost_history[400])
print(cost_history[-1])

# ploting a chart of costfunction X epochs.
plot.plot(cost_history,"b")
plot.show()

