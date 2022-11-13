from itertools import combinations
feature_emb_list=['a','b','c','d','e']
# for i, v in enumerate(combinations(feature_emb_list, 2)):
#     print(i)
#     print(v)

for i, j in combinations(range(len(feature_emb_list)), 2):
    print("i",i)
    print("j",j)

# for v_i, v_j in combinations(feature_emb_list, 2):
#     print("i",v_i)
#     print("j",v_j)