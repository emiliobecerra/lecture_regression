import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")

#how to shuffle your dataset
dataset.sample(frac=1)

# print(dataset)
target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:9].values

#We're chopping our data into 4 parts.
kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

# print(kfold_object)

i=0
for train_index, test_index in kfold_object.split(data):
	i=i+1
	print("Round:", str(i))
	print("Training index: ")
	print(train_index)
	print("Test index: ")
	print(test_index)

	data_train = data[train_index]
	target_train = target[train_index]
	data_test = data[test_index]
	target_test = target[test_index]

	machine = linear_model.LinearRegression()
	machine.fit(data_train, target_train)

	prediction = machine.predict(data_test)

	#Let's find out if these two variations explain each other well. Is prediction is explaining the target test?
	#In other words, the r-squared of prediction and target test

	r2 = metrics.r2_score(target_test, prediction)
	print("R square score: ", r2)
	print("\n\n")