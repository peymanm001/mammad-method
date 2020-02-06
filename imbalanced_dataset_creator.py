import numpy as np
import pandas as pd
from sklearn import preprocessing

dataset_path = './datasets'

glass_path = dataset_path + "/Glass.txt"
ecoli_path = dataset_path + "/ecoli.csv"
# glass = pd.read_csv(glass_path)

# df = pd.read_csv(ecoli_path, header=-1).values
#
# # ecoli
# X = df[:, 1:-1]
# y = df[:, -1]
# le = preprocessing.LabelEncoder()
# le.fit(y)
# y = le.transform(y)
#
# np.save(dataset_path + "/ecoli_test", df)
#
df = np.load(dataset_path + "/ecoli_test.npy", allow_pickle=True)
# df[df == 'cp'] = 0
# df[df == 'im'] = 1
# df[df == 'imS'] = 2
# df[df == 'imL'] = 3
# df[df == 'imU'] = 4
# df[df == 'om'] = 5
# df[df == 'omL'] = 6
# df[df == 'pp'] = 7
# np.save(dataset_path + "/ecoli_test", df)
print(np.unique(df[:, 0]))
print(df)
pass
