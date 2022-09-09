import pandas as pd

data_df = pd.read_table('E:/DataSet/train_1m.txt')

columns=[]
for i in range(1,41):
    columns.append("Column"+str(i))

print(columns)
data_df.columns = columns
data_df.to_csv('E:/DataSet/train.csv', index=0)