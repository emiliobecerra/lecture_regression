#An optimization
import pandas
from sklearn import linear_model

import kfold_template

dataset = pandas.read_csv("dataset.csv")

dataset = dataset.sample(frac=1)

target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:9].values

machine = linear_model.LinearRegression()

kfold_template.run_kfold(machine, data, target, 4, True)