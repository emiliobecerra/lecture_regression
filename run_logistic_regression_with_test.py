import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")

#how to shuffle your dataset
dataset.sample(frac=1)

# print(dataset)
target = dataset.iloc[:,1].values
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

	machine = linear_model.LogisticRegression()
	machine.fit(data_train, target_train)

	prediction = machine.predict(data_test)

	#

	accuracy_score = metrics.accuracy_score(target_test, prediction)
	print("Accuracy score: ", accuracy_score)
	confusion_matrix = metrics.confusion_matrix(target_test, prediction)
	print("Confusion matrix: ")
	print(confusion_matrix)
	print("\n\n")