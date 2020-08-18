#importando bibliotecas necessarias
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#carregamento do dataset
from sklearn.preprocessing import LabelEncoder

url = 'ks-projects-201801.csv'
names = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'state', 'backers', 'country', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']
dataset = pandas.read_csv(url)

print('Shape dos dados')
print(dataset.shape)

print('Visualizando o conjunto inicial dos dados:')
print(dataset.head(20))

print('Conhecendo os dados estatisticos do dataset:')
print(dataset.describe())

print('Conhecendo a distribuicao dos dados por classes:')
print(dataset.groupby('state').size())

#transforma categoricos em numericos
le = LabelEncoder()
for index, item in enumerate(dataset.columns):
    if index in range(15):
        #print(dataset[item])

        dataset[item] = le.fit_transform(dataset[item].astype(str))
    else:
        pass

print('Criando graficos de caixa da distribuicao das classes')
dataset.plot(kind='box', subplots=True, layout=(3,5), sharex=False, sharey=False)
plt.title('Graficos de caixa da distribuicao das classes')
plt.show()

print('Criando histogramas dos dados por classe')
dataset.hist()
plt.title('Histogramas dos dados por classe')
plt.show()

print('Criando graficos de dispersao dos dados')
axs = scatter_matrix(dataset)
plt.title('Graficos de dispersao dos dados')
#plt.savefig('grafico de dispersao.png')
plt.show()
