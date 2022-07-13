
# JONAS NUNES
# https://github.com/joneco02
# Perceptron Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading dataset
data = pd.DataFrame(pd.read_csvcsv('F:/Users/Joneco/Documents/redes neurais/perceptron/ex1traindata1.csv'))

#plotting data set with different symbols for each class
#Inputs
X = data[['X1','X2']].values
#Known Outputs
y = data['labels'].values

Group0 = data[data['labels']<1]
Group1 = data[data['labels']>0]

plt.figure(figsize=(12,12))
p1=plt.scatter(Group0["X1"],Group0["X2"],marker="*",label= "Group 0", s = 200)
p2=plt.scatter(Group1["X1"],Group1["X2"],marker="$\clubsuit$",label= "Group 1", s = 200)
plt.title('Data Plot'), plt.xlabel("X1"), plt.xlabel("X2")
plt.legend()
plt.show()

#Table with some statistics of each class
media_geral=pd.Series(data[['X1','X2']].mean())
desvio_geral=data[['X1','X2']].std()
max_geral=data[['X1','X2']].max()
min_geral=data[['X1','X2']].min()

est_geral = pd.DataFrame({'Mean':media_geral, 'Standard Deviation': desvio_geral, 'Max. Value':max_geral,'Min. Value':min_geral}).T
print('Statistics for all')
est_geral.style

#Statics for Each Group
#Statics for Group 
total_mean=pd.Series(Group0[['X1','X2']].mean())
total_std_deviation=Group0[['X1','X2']].std()
total_max=Group0[['X1','X2']].max()
total_min=Group0[['X1','X2']].min()

est_Group0 = pd.DataFrame({'Mean':total_mean, 'Standard Deviation': total_std_deviation, 'Max. Value':total_max,'Min. Value':total_min}).T
print('Statistics Group 0')
est_Group0.style

#Statics for Group 1
total_mean=pd.Series(Group1[['X1','X2']].mean())
total_std_deviation=Group1[['X1','X2']].std()
total_max=Group1[['X1','X2']].max()
total_min=Group1[['X1','X2']].min()

est_Group1 = pd.DataFrame({'Mean':total_mean, 'Standard Deviation': total_std_deviation, 'Max. Value':total_max,'Min. Value':total_min}).T
print('Statistics Group 1')
est_Group1.style

#Creating raw functions to define a perceptron network
def prediction(X):
    # u = somatorio_i^n (wi * xi + bias)
    u = np.dot(X, weights[1:]) + weights[0]*1
    return (u>0)*1

def weights_training(X, y,lr,epochs,weights):
    
    #for each epoch
    for i in range(epochs):
        # this for is to define the sum range
        for i in range(1,len(X),1):
            d = prediction(X[i])
            #lr = learning rate, yi = output
            weights[1:] += lr * (y[i] - d) * X[i]
            weights[0] += lr * (y[i] - d) *1
    return weights

def accuracy_calc(X,y):
    acc=0
    size=y.size
    d=prediction(X)
    return ((y==d)*1).sum()/size


#doing a search to find the best learning rate and epochs number for the perceptron network using random values as initial weights value
rep=30
epochs=[10,20,50,100,200]
#lenght of the weights vector
size = 3
table = pd.DataFrame(columns=['LR','epochs','Media','Desvio','weights'])

#lrates = [0.01,0.03]
lrates=[0.01, 0.03, 0.1, 0.3, 1]

accuracy = []

for lr in lrates:
    for epoca in epochs:
        
        for i in range(rep):
            weights = np.random.rand(tamanho)
            weights_training(X, y,lr,epoca,weights)
            accuracy.append(accuracy_calc(X,y)) # Adds calculated accuracy to the end of the list
        mean = np.array(accuracy).mean()
        standard_deviation = np.array(accuracy).std()
        table = table.append({'LR':lr,'epochs':epoca,'Mean':mean,'Standard Deviation':standard_deviation, 'weights':weights},ignore_index=True)
        
#Print Table
table.style

epochs = 30
lr = 0.01
weights = np.random.rand(tamanho)
weight0 = np.array(weights_training(X, y,lr,epochs,weights))
weights = np.random.rand(tamanho)
weight1= np.array(weights_training(X, y,lr,epochs,weights))
weights = np.random.rand(tamanho)
weight2= np.array(weights_training(X, y,lr,epochs,weights))
weights = np.random.rand(tamanho)
weight3= np.array(weights_training(X, y,lr,epochs,weights))
weights = np.random.rand(tamanho)
weight4= np.array(weights_training(X, y,lr,epochs,weights))

x1=np.arange(0,9,0.5)
x2_0=(-weight0[0]-weight0[1]*x1)/weight0[2]
x2_1=(-weight1[0]-weight1[1]*x1)/weight1[2]
x2_2=(-weight2[0]-weight2[1]*x1)/weight2[2]
x2_3=(-weight3[0]-weight3[1]*x1)/weight3[2]
x2_4=(-weight4[0]-weight4[1]*x1)/weight4[2]

plt.figure(figsize=(15,15))

plt.title("Tarefa 1"), plt.xlabel("X1"), plt.title("Tarefa 1"), plt.xlabel("X1"), 
plt.scatter(Group0["X1"],Group0["X2"],marker="*", alpha=0.5,s=400, label='Group 0'), plt.scatter(Group1["X1"],Group1["X2"],marker="$\clubsuit$", alpha=0.5, s=400, label='Group 1')

plt.plot(x1,x2_0, label='Weight 0')
plt.plot(x1,x2_1, label='Weight 1')
plt.plot(x1,x2_2, label='Weight 2')
plt.plot(x1,x2_3, label='Weight 3')
plt.plot(x1,x2_4, label='Weight 4')

plt.legend()
plt.show()

#Conclusions and Results

# Considering that all the attempts of the table gave an accuracy of 100% (Mean 1 and deviation 0), the best would be the one that used the least resources.
# So it would be the one with the lowest epoch, = 10 and the one that would use the most abstraction, so it would need "less training", then it would be the one with the highest LR.
# The best option FOR THIS CASE would be 10 epochs and Learning Rate = 1
# When we vary the number of epochs and the learning rate, we will have a different set of weights in the experiment.


#testing the network prediction for the last weights (purple line in the diagram above), it will print 0 if the data is in Group0 and 1 if data is in Group1

prediction([8,2.5])

#Visual Checking point position with weight and group/segregation line.
plt.plot(x1,x2_4, label='Weight 4')
plt.scatter(8,2.5)
plt.text(4, 12.5, "Group 0", size=20, rotation=+30.,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
plt.text(6, 6.5, "Group 1", size=20, rotation=+30.,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
plt.show()