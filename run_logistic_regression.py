#We need logistic regression when the Y variable is limited from (1 , 0). 
#Let's regress y2
import pandas

from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")

print(dataset)

#we want to turn dataset into matrix form. 

#dataset.iloc[row,column]
target = dataset.iloc[:,1].values
#.values makes it into matrix, not vector
data = dataset.iloc[:,3:9].values

#Here is what we have:  dataset     new_dataset
#               target-  we have        ?
#               data -   we have      we have
# we make a prediction about ?

#Let's make a machine to train
machine = linear_model.LogisticRegression()
machine.fit(data,target)

new_dataset = pandas.read_csv("new_dataset.csv")
new_dataset = new_dataset.values

prediction = machine.predict(new_dataset)

print(prediction)

#for multiple categorical variables, we cannot use linear regression.
# Instead you do a multinomial logit regression. It's not the most efficient.
# It doesn't take into account the order. 
# So, do order logit through scikit extend package. 

#For count data (non-negative integers) do poisson: PoissonRegression