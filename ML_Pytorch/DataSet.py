import pandas as pd
import Word2VecPreprocessing
import numpy
from gensim.models import Word2Vec
data_df = pd.read_csv('E:/DataSet/data.csv')

# columns=[]
# for i in range(1,41):
#     columns.append("Column"+str(i))
#
# print(columns)
# data_df.columns = columns
# data_df.to_csv('E:/DataSet/train.csv', index=0)
# Word2Vec.train_vec()
# for i in range(15,41):
# print(len(data_df['Column15'].unique()))

column = data_df['Column15'].unique()
# f=open("./Coulum15.txt","w")
#
# for c in column:
#     f.write(c+"\n")
# f.close()

# column.to_csv('./Coulum15.txt', index=0)
# Word2VecPreprocessing.train_vec([list])
vector_file = r'.\zh_wiki.model'
model = Word2Vec.load(vector_file)
print(model.wv.vector_size)
print(column[0],model.wv[column[0]])
print(column[1],model.wv[column[1]])