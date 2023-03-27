import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy as sci
import matplotlib
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from fcmeans import FCM
import scikit_posthocs as sci_posthocs

subj_data = pd.read_csv('../data/subj_description.txt', index_col=0)

def load_data(path):
    brain_data = pd.read_csv(path +'/brain_data.txt', index_col=0)
    cog_data = pd.read_csv(path +'/cog_data.txt', index_col=0)
    subj_data = pd.read_csv(path +'/subj_description.txt', index_col=0)
    return brain_data, cog_data, subj_data


def get_group_data(data, group, brain=False, corr=False, subj=False):
    if brain:
        data = data.loc[subj_data['phenotype_description']==group]
    else:
        data = data.drop(columns=['sex', 'interview_age', 'sestot', 'mot_edscale']).loc[subj_data['phenotype_description']==group]
    if subj:
        data_corr = data.transpose()
    else: 
        data_corr = data
        
    if corr:
        return data_corr.corr()
    
    return data

def get_cluster_data(data, k):
    return data.loc[data['cluster']==k]

def removenan(array):
    nan_array = np.isnan(array)
    not_nan_array = ~ nan_array
    array_ = array[not_nan_array]
    return array_

def permute_df(df):
    df_p = pd.DataFrame()
    col = np.array(df.columns)
    for c in col: 
        df_p[c] = np.random.permutation(df[c])
    return df_p

def permutation_test(data, num_simulations):
    var_p = []
    for i in range(num_simulations):
        data_perm = permute_df(data)
        pca_p = PCA()
        pca_p.fit(data_perm)
        var_p.append(pca_p.explained_variance_ratio_)
    return np.array(var_p)

def group_comparison(data, col_df, cluster=False, control=False):
    if cluster:
        data_0 = get_cluster_data(data, 0)
        data_1 = get_cluster_data(data, 1)
        data_2 = get_cluster_data(data, 2)
        df_comparison = pd.DataFrame({'test': ['group', '0-1', '0-2', '1-2']}).set_index('test')
        
    else:
        data_0 = get_group_data(data, 0)
        data_1 = get_group_data(data, 1)
        data_2 = get_group_data(data, 2)
    
        df_comparison = pd.DataFrame({'test': ['group', 'contr - affect.', 'contr - non-aff.', 'affect - non-aff.']}).set_index('test')

    for column in col_df:
        comp = []
        group = sci.stats.kruskal(removenan(data_0[column].values), 
                                  removenan(data_1[column].values), 
                                  removenan(data_2[column].values))
        comp.append([round(group[0],4), round(group[1],4)])

        if group[1] <= 0.05:
            posthoc = sci_posthocs.posthoc_dunn([data_0[column], data_1[column], data_2[column]])
            comp.append([round(sci.stats.kruskal(removenan(data_0[column].values), removenan(data_1[column].values))[0],4), round(posthoc[1][2],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_0[column].values), removenan(data_2[column].values))[0],4), round(posthoc[1][3],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_1[column].values), removenan(data_2[column].values))[0],4), round(posthoc[2][3],4)])

        else:
            comp.extend(['-', '-', '-'])

        df_comparison[column] = comp
    return df_comparison

def group_comparison_control(data, col_df):
    data_0 = get_cluster_data(data, 0)
    data_1 = get_cluster_data(data, 1)
    data_2 = get_cluster_data(data, 2)
    data_c = get_cluster_data(data, 'c')
    df_comparison = pd.DataFrame({'test': ['group', '0-1', '0-2', '0-c', '1-2', '1-c', '2-c']}).set_index('test')

    for column in col_df:
        comp = []
        group = sci.stats.kruskal(removenan(data_0[column].values), 
                                  removenan(data_1[column].values), 
                                  removenan(data_2[column].values),
                                  removenan(data_c[column].values))
        comp.append([round(group[0],4), round(group[1],4)])

        if group[1] <= 0.05:
            posthoc = sci_posthocs.posthoc_dunn([data_0[column], data_1[column], data_2[column], data_c[column]])
            comp.append([round(sci.stats.kruskal(removenan(data_0[column].values), removenan(data_1[column].values))[0],4), round(posthoc[1][2],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_0[column].values), removenan(data_2[column].values))[0],4), round(posthoc[1][3],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_0[column].values), removenan(data_c[column].values))[0],4), round(posthoc[1][4],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_1[column].values), removenan(data_2[column].values))[0],4), round(posthoc[2][3],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_1[column].values), removenan(data_c[column].values))[0],4), round(posthoc[2][4],4)])
            comp.append([round(sci.stats.kruskal(removenan(data_2[column].values), removenan(data_c[column].values))[0],4), round(posthoc[3][4],4)])

        else:
            comp.extend(['-', '-', '-', '-', '-', '-'])

        df_comparison[column] = comp
    return df_comparison

def get_feature_importance(data, var, components):
    df_components = pd.DataFrame(abs(components.T))
    df_components.columns = [''.join(['PC', f'{i+1}']) for i in range(components.shape[0])]
    features = list(data.columns)
    df_components.index = features
    df_components = df_components*var
    df_components = df_components.sort_values(by=list(df_components.columns), ascending=False)
    return df_components

def get_highest_contr_feat(df_components, PC, i=10):
    return df_components.sort_values(PC, ascending=False).index[:i]

def get_intertia(data, max_cluster):
    K = range(2,max_cluster+1)
    inertia = []
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        inertia.append(kmeanModel.inertia_)
    return K, inertia

def get_partition(data, max_cluster):
    K = range(2,max_cluster+1)
    models = list()
    for n_clusters in K:
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(data)
        models.append(fcm)

    num_clusters = len(K)
    pc_list = []
    pec_list = []
    for n_clusters, model in zip(K, models):
        # get validation metrics
        pc_list.append(model.partition_coefficient)
        pec_list.append(model.partition_entropy_coefficient)
    return K, pc_list, pec_list

def calculate_cluster_stats(results, subj_data, patients_only=False):
    comparison = pd.DataFrame(subj_data['phenotype_description'])
    comparison['cluster'] = results
    df = comparison.groupby(['phenotype_description', 'cluster']).size().unstack(fill_value=0)
    if not patients_only:
        df.loc[0] = df.loc[0].div(46)  # control
    df.loc[1] = df.loc[1].div(25)  # affective
    df.loc[2] = df.loc[2].div(84)  # nonaffective
    return df

def K_Means(data, k, subj_data=subj_data, patients_only=False):
    kmeans_model = KMeans(n_clusters=k)  # , random_state=2023
    kmeans_predict = kmeans_model.fit_predict(data)
    centers = kmeans_model.cluster_centers_
    return kmeans_predict, calculate_cluster_stats(kmeans_predict, subj_data, patients_only)
    
def Spectral(data, k, subj_data=subj_data, patients_only=False):
    model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')  # ,  random_state=2023
    spectral_predict = model.fit_predict(data)
    return spectral_predict, calculate_cluster_stats(spectral_predict, subj_data, patients_only)
    
def Fuzzy(data, k, subj_data=subj_data, patients_only=False, random_state=2023):
    model = FCM(n_clusters=k, random_state=random_state) 
    model.fit(data) 
    centers = model.centers
    fuzzy_predict = model.predict(data)

    return fuzzy_predict, calculate_cluster_stats(fuzzy_predict, subj_data, patients_only)


