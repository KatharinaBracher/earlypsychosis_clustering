import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

rc('text', usetex=False)
font = {#'family':'CMU Serif', 
               #'sans-serif':['Helvetica'],
               'size': 11}
#mpl.rc('font', **font)

newcolors = ['#3880d9', '#910040', '#ff603d'] #b2182b #ef8a62 #fddbc7 #d1e5f0 #67a9cf #2166ac
pheno_col = ListedColormap(newcolors, name='my')
newcolors = ['#910040', '#ff603d']
pheno_pat_col = ListedColormap(newcolors, name='my')


# newcolors2 = ['#0c2c84', '#afc342', '#617e49', '#64b09e'] 
newcolors2 = ['#afc342','#64b09e', '#0c2c84' ]  #['#0c2c84', '#afc342', '#64b09e'] 
cluster_col = ListedColormap(newcolors2, name='my2')

###############################################################
subj_data = pd.read_csv('../data/subj_description.txt', index_col=0)


def plot_PC(data, var, var_p_mean, index_significant):
    fig, ax = plt.subplots(figsize=(2.5,1.3))
    feat = data.shape[1]
    ax.plot([f"PC{i+1}" for i, v in enumerate(var)], var * 100, linestyle='-', color='k', linewidth=1, marker='o', markersize=2)
    ax.plot([f"PC{i+1}" for i, v in enumerate(var)], var_p_mean * 100, linestyle='--', color='grey', linewidth=2)
    ax.plot([f"PC{i+1}" for i, v in enumerate(var)][:index_significant+1], var[:index_significant+1] * 100, linestyle='-', color='blue', linewidth=2, marker='o', markersize=4)

    ax.text(0.15, 1, 'PCs', transform=ax.transAxes, verticalalignment='top', weight='bold')
    textstr = '\n'.join((
        '1-'+str(index_significant+1)+'***',
        str(index_significant+2) +'-'+str(feat)))
    props = dict(boxstyle='square', facecolor='blue', alpha=0.3)
    ax.text(0.15, 0.82, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_ylabel('explained variance [%]')
    label = textwrap.fill('explained variance [%]', width=15,
                  break_long_words=True)
    ax.set_ylabel(label,fontsize=9.5)

    ax.set_xticks(np.arange(0,feat,10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def plot_data_2D(data_pca, patients = False, group=subj_data, col=pheno_col, cluster = False, FA=False):

    if patients:
        grouping = group[group.phenotype != 0]['phenotype_description']
        col = pheno_pat_col
        labels = ['affective', 'non-affective']
    elif cluster:
        col = cluster_col
        labels = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']
        grouping = group
    else:
        labels = ['controls', 'affective', 'non-affective']
        grouping = group['phenotype_description']
    
    axlabels = ['PC1', 'PC2']
    if FA:
        axlabels = ['F1', 'F2']
    
    fig, ax = plt.subplots(figsize=(1.5,1)) #big 2,1.5

    scatter = ax.scatter(data_pca[:,0], data_pca[:,1], c=grouping, cmap=col, alpha=0.6, s=15, lw=0)
    lgnd = ax.legend(handles=scatter.legend_elements()[0], labels=labels, loc='center left', bbox_to_anchor=(0.9, 0.5), frameon=False)
    ax.set_xlabel(axlabels[0])
    ax.set_ylabel(axlabels[1])
    #ax.set_yticks([0, 10])
    #ax.set_xticks([0,10])
    #ax.tick_params(labelsize=18)

    #change the marker size manually for both lines
    #lgnd.legendHandles[0]._legmarker.set_markersize(5)
    #lgnd.legendHandles[1]._legmarker.set_markersize(5)
    #if not patients:
    #    lgnd.legendHandles[2]._legmarker.set_markersize(5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def plot_data_3D(data_pca, patients = False, group=subj_data, pheno_col=pheno_col):
    labels = ['control', 'affective', 'non-affective']
    if patients:
        group = group[group.phenotype != 0]
        pheno_col = pheno_pat_col
        labels = ['affective', 'non-affective']
    
    fig = plt.figure(figsize=(2,3))
    ax = fig.add_subplot(projection='3d')
    #1,2,0
    ax.scatter(data_pca[:,2],data_pca[:,1],-data_pca[:,0], c=group['phenotype_description'], cmap=pheno_col, s=15, lw=0, alpha=0.6)

    ax.set_xlabel('PC 3', labelpad=2)
    ax.set_ylabel('PC 2', labelpad=2)
    ax.set_zlabel('-PC 1', labelpad=1)

    #ax.set_xticks([-10, -5, 0,5])
    #ax.set_yticks([-7.5,0,7.5])
    #ax.set_zticks([-4,0,4])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    
def plot_clustering_validation(K, inertia, label='inertia'):
    fig, ax = plt.subplots(figsize=(2,1))
    ax.plot(K, inertia, '-', c='k', linewidth=2,  marker='o', markersize=4)
    ax.set_xlabel('# cluster')
    ax.set_ylabel(label)
    ax.set_xticks([2,4,6,8])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def plot_cluster_result(df, number=3, patients_only=False):
    fig, ax = plt.subplots(1, 1, figsize=(1.5,1.2)) #1.2, 0.8   1.2,1.2

    g0 = sns.heatmap(df, cmap='Blues',cbar=False, vmin=0, vmax=1, annot=True, fmt=".0%", annot_kws={'size': 10}, yticklabels=True)
    g0.set_ylabel('')
    g0.set_xlabel('')
    if patients_only:
        g0.set_yticklabels(['affective', 'non-affective'], rotation='horizontal')
    else:
        g0.set_yticklabels(['control', 'affective', 'non-affective'], rotation='horizontal')
    if number==3:
        g0.set_xticklabels(['0', '1', '2'])
    elif number==4:
        g0.set_xticklabels(['0', '1', '2', '3'])
    else:
        pass
    g0.set_title('Cluster')


def plot_boxplots(data, columns, col_names, m=2,n=5, control=False):
    pal = ['#0c2c84', '#afc342', '#64b09e']#, '#64b09e']
    order = [0, 1, 2]#, 3]
    
    if control:
        pal = ['#0c2c84', '#afc342', '#64b09e', '#3880d9'] #'#0c2c84', '#64b09e', '#bfcf69','#3880d9'
        order = [0, 1, 2, 'c']

    fig, ax = plt.subplots(m,n, figsize=(4.9,1.8)) #  6.5,2 ; 5.,2 ;  9,4
    
    for i,col in enumerate(columns):
        g0 = sns.boxplot(x='cluster', y=col, data=data, palette=pal, order=order, ax=ax.flatten()[i])
        sns.stripplot(x='cluster', y=col, data=data, color='.25', size=3, order=order, ax=ax.flatten()[i])
        #g0.set(ylabel=col_names[i])
        label = textwrap.fill(col_names[i], width=20,
                      break_long_words=True)
        g0.set_ylabel(label,fontsize=9.5)
        g0.set(xlabel='')
        g0.set_ylim(-0.1, 0.12)
        
        '''if i>n*m-(n+1):
            g0.set(xlabel='Cluster')
        g0.set_ylim(data[col].min()-5, data[col].max()+data[col].max()*0.2)'''
        
        if col=='deldisk01_auc_40000':
            g0.set_ylim(data[col].min()-0.2, data[col].max()+data[col].max()*0.9)
        if col=='acpt01_auditory_t14':
            g0.set_ylim(data[col].min()-0.2, data[col].max()+data[col].max()*0.5)
        if col=='pcps01_nih_patterncomp_ageadjusted':
            g0.set_ylim(data[col].min()-0.2, data[col].max()+data[col].max()*0.5)
        if col=='wasi201_matrix_totalrawscore':
            g0.set_ylim(data[col].min()-0.2, data[col].max()+data[col].max()*0.5)
    
    plt.tight_layout()