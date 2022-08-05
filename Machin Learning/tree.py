import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

tr_sample = tree.DecisionTreeClassifier()
lb_sample = LabelEncoder()

inf = pd.read_csv('tree.csv')

all_info = inf.drop('salary' , axis='columns')
price = inf.salary

all_info['factory'] = lb_sample.fit_transform(all_info['factory'])
all_info['job'] = lb_sample.fit_transform(all_info['job'])
all_info['degree'] = lb_sample.fit_transform(all_info['degree'])

tr_sample.fit(all_info , price)

tr_sample.predict([[2,1,0]])