import numpy as np
import pandas as pd


def calculate_new_vec(vec_trong_so, vec, learn):
    a = list()
    for i in range(len(vec_trong_so)):
        new_data = vec_trong_so[i] + learn*(vec[i] - vec_trong_so[i])
        a.append(new_data)
    new_vec = np.array(a)
    return new_vec


def cluster(list_vect_trong_so, list_vect):
    cluster = {}
    for i in range(len(list_vect)):
        list_dist = list()
        for j in range(len(list_vect_trong_so)):
            distance = np.linalg.norm(list_vect[i] - list_vect_trong_so[j])
            list_dist.append(distance)
        min_index = list_dist.index(min(list_dist))
        # temp = list([i])
        if min_index + 1 not in cluster.keys():
            a = [i + 1]
            cluster[min_index + 1] = a
        else:
            a = cluster[min_index + 1]
            a.append(i + 1)
            cluster[min_index + 1] = a
    return cluster


def kohonen():
    df = pd.read_csv('./Data.csv')
    v1 = np.array([11, 55, 200])            # khoi tao vector trong so
    v2 = np.array([15, 10, 300])            # thay vector trong so vao day
    v3 = np.array([18, 20, 600])
    vector_trong_so = [v1, v2, v3]
    epochs = 10
    r = 0
    learn = 0.8
    # print(df.head())
    df_cp = df.copy()
    df_cp.drop(columns=['Vị trí'], inplace=True)
    # print(df_cp.head())
    list_vector = df_cp.values
    for epoch in range(epochs):
        for i in range(df_cp.shape[0]):
            list_dis = list()
            for j in range(len(vector_trong_so)):
                distance = np.linalg.norm(list_vector[i] - vector_trong_so[j])
                list_dis.append(distance)
                # print('D', i + 1, j + 1, '=', distance)
            min_dist_vec_index = list_dis.index(min(list_dis))
            vector_trong_so[min_dist_vec_index] = calculate_new_vec(vector_trong_so[min_dist_vec_index], list_vector[i], learn)
        print('Lan lap', epoch, 'Vector trong so', vector_trong_so)
        learn = learn/2
    clust = cluster(vector_trong_so, list_vector)
    print('------------------------------------------------')
    print('Gom cum:')
    print(clust)


kohonen()
