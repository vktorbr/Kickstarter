import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

#carregamento dos dados
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

url = 'ks-projects-201801.csv'
dataset = pd.read_csv(url)
#print(dataset.head(20))

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(dataset.corr(), annot = True, fmt = '.2f', cmap='Blues', linewidths=0.02, square=True, cbar=False, ax = ax)
plt.yticks(rotation='horizontal')
plt.xticks((np.arange(7)), ('ID', 'Goal', 'Pledged', 'backers', 'usdP', 'usdPReal', 'usdGoal'))
plt.title('Correlação entre variáveis do dataset')
plt.savefig('correlacao.png')
plt.show()

array = dataset.values
y = array[:, 9]
print(y[:10])
#transforma categoricos em numericos
le = LabelEncoder()
for index, item in enumerate(dataset.columns):
    if index in range(15):
        #print(dataset[item])

        dataset[item] = le.fit_transform(dataset[item].astype(str))
    else:
        pass

#feature importance
array = dataset.values
y = array[:, 9]
print(y[:10])
dataset = dataset.drop('state', axis=1)
X = dataset.values

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier(n_estimators=10, random_state=10)
model.fit(X_train, y_train)# Mostrando importância de cada feature
importances = pd.Series(data=model.feature_importances_, index=dataset.columns)
sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importancia de cada feature')
plt.savefig('importancia.png')
plt.show()

headers = ["name", "score"]
values = sorted(zip(dataset.columns, model.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))

