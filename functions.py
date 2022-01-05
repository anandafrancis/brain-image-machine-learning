#%%
# import necessary libraries

from nilearn import plotting
from nilearn import image
from nilearn.image import get_data
from nilearn import datasets
from nilearn import input_data
from nilearn.connectome import ConnectivityMeasure

from sklearn import metrics
from sklearn import preprocessing

import pandas as pd
import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier 
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
#%%

def get_filename(sub, emotion, priming):
    '''
    Purpose: 
        obtain file names from dataset folder (function must run in same directory as files)
    
    Parameters:
        sub : int
            Number of subject
            
        emotion : str
            Emotion state (anger, fear, disgust)
            
        priming : str
            Priming conditions (congruent,, incongruent, neutral)
    
    Returns: 
        Filename
    '''
    
    # get the subject as a string
    if sub // 10 == 0:
        sub_str = "00" + str(sub)
    
    else:
        sub_str = "0" + str(sub)
    
    # create the filename
    return 'cp' + sub_str + '_beta_' + emotion + '_' + priming + '_c0.nii.gz'

#%%

def create_file_list(subs, emotions, primings):
    '''
    Purpose: 
        Create list of file names of specified subjects, emotion state and priming conditions
    
    Parameters:
        subs : list
            List of subject numbers
            
        emotions : list
            List of emotion states (anger, fear, disgust)
            
        priming : list
            List of priming conditions (congruent,, incongruent, neutral)
    
    Returns: 
        List of file names representing list of brain image files
    '''
    
    # add every combination of subject number, emotion state and priming condition to list
    return [get_filename(sub, emo, prime) for sub in subs for emo in emotions for prime in primings]

#%%

def get_class(file):
    '''
    Purpose: 
        get the emotion state of a file based on the name of the file
    
    Parameters:
        file : str
            File name
    
    Returns: 
        emotion state
    '''
    
    # define classes
    a = 'anger'
    d = 'disgust'
    f = 'fear'
    
    # create list of classes
    classes = [a, d, f]
    
    # iterate through list to find class
    for class_ in classes:
        if class_ in file:
            file_class = class_
    
    # output class of file
    return file_class

#%%

def get_priming(file):
    '''
    Purpose: 
        get the priming condition of a file based on the name of the file
    
    Parameters:
        file : str
            File name
    
    Returns: 
        priming condition
    '''
    
    # define priming conditions
    c = '_congruent'
    i= '_incongruent'
    n = '_neutral'
    
    # create list of classes
    primings = [c, i, n]
    
    # iterate through list to find class
    for priming in primings:
        if priming in file:
            file_priming = priming
    
    # output class of file
    return file_priming[1:]


#%%

def plot_sub_emotion(sub, emotion, coordinates = (0, 0, 0)):
    '''
    Purpose: 
        obtains subject and emotion and plots the
        brain under the 3 different priming conditions
    
    Parameters:
        sub : int
            Number of subject
        
        emotion : str
            Name of emotion = 'anger', 'fear', 'disgust'
            
        coordinates : tuple (optional, default = (0, 0, 0))
            X, Y and Z values for plot 
    
    Returns: 
        none, just renders 3 plots
    '''
    
    # get the brain image file from each dataset
    sub_emotion_c = get_filename(sub, emotion, 'congruent')
    sub_emotion_i = get_filename(sub, emotion, 'incongruent')
    sub_emotion_n = get_filename(sub, emotion, 'neutral')

    # create a title
    title_base = 'Sub ' + str(sub) + ' ' + emotion
    
    # customize title based on priming condition
    n_title = title_base + ' x neutral'
    c_title = title_base + ' x congruent'
    i_title = title_base + ' x incongruent'
    
    
    # plot each brain visualization
    plotting.plot_roi(sub_emotion_c, title = c_title, cut_coords = coordinates, black_bg=True)
    plotting.plot_roi(sub_emotion_i, title = i_title, cut_coords = coordinates, black_bg=True)
    plotting.plot_roi(sub_emotion_n, title = n_title, cut_coords = coordinates, black_bg=True)
    
#%%

def euclidean(p1, p2):
    '''
    Purpose: 
        Euclidean distance measure
    
    Parameters:
        p1 : list
            Vector 1
            
        p2 : list
            Vector 2
        
    Returns: 
        euclidean distance 
    '''
    
    return sum([(x1-x2)**2 for x1,x2 in zip(p1,p2)]) ** 0.5


#%%

def mag(v):
    '''
    Purpose: 
        calculate magnitude of a vector
    
    Parameters:
        v : list
            Vector
        
    Returns: 
        magnitude of vector
    '''
    return sum([i **2 for i in v]) ** 0.5

#%%

def dot(u,v):
    '''
    Purpose: 
        calculate dot product of two vectors
    
    Parameters: 
        u : list
            Vector 1
        v : list
            Vector 2
        
    Returns: 
        dot product of the 2 vectors
    '''
    return sum([ui * vi for ui, vi in zip(u,v)])

#%%

def cosine_similarity(u, v):
    '''
    Purpose: 
        calculate cosine similiarity between vectors
    
    Parameters:
        u : list
            Vector 1
        v : list
            Vector 2
    
    Returns: 
        costine similarity between the 2 vectors
    '''
    
    return dot(u,v)/(mag(u) * mag(v))
    
#%%

def plot_mean_imgs(averaging, condition, coordinates = (0, 0, 0)):
    '''
    Purpose:
        Plot average brain images between list of brains
        
    Parameters:
    
        averaging : str
            Method of which images will be averaged over
            
        condition : str or int
            Attribute that images will be averaged over
            
        coordinates : tuple (optional, default = (0, 0, 0))
            X, Y and Z values for plot
            
    Returns:
        None, just renders mean img plots
    '''
    
    subs = list(range(1, 31))
    emotions = ['anger', 'disgust','fear']
    primings = ['congruent', 'incongruent', 'neutral']
    
    if averaging == 'emotion':
            images = create_file_list(subs, [condition], primings)
            title = condition + ' mean image'
            
    if averaging == 'priming':
            images = create_file_list(subs, emotions, [condition])
            title = condition + ' mean image'

    if averaging == 'subject':
            images = create_file_list([condition], emotions, primings) 
            title = 'subject ' + str(condition) + ' mean image'
            
    mean_image = image.mean_img(images)        
    plotting.plot_roi(mean_image, title= title, cut_coords = coordinates, black_bg= True)
            
#%%
def parcellation_atlas_dict():
    
    '''
    Purpose:
        Create a dictionary of all parcellation techniques and their maps and labels
    Parameters:
       None

    
    Returns: dict 
       Dictionary of all the brain parcellation atlas maps in the NiLearn dataset  
    '''

    
    # Harvard Oxford Atlases
    harvard_oxford_cort_0_1 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
    harvard_oxford_cort_0_2 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm')
    harvard_oxford_cort_25_1 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    harvard_oxford_cort_25_2 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    harvard_oxford_cort_50_1 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
    harvard_oxford_cort_50_2 = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-2mm')
    harvard_oxford_cort_1 = datasets.fetch_atlas_harvard_oxford('cort-prob-1mm')
    harvard_oxford_cort_2 = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
    harvard_oxford_cortl_0_1 = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr0-1mm')
    harvard_oxford_cortl_0_2 = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr0-2mm')
    harvard_oxford_cortl_25_1 = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-1mm')
    harvard_oxford_cortl_25_2 = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-2mm')
    harvard_oxford_cortl_50_1 = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr50-1mm')
    harvard_oxford_cortl_50_2 = datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr50-2mm')
    harvard_oxford_cortl_1 = datasets.fetch_atlas_harvard_oxford('cortl-prob-1mm')
    harvard_oxford_cortl_2 = datasets.fetch_atlas_harvard_oxford('cortl-prob-2mm')
    harvard_oxford_sub_0_1 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-1mm')
    harvard_oxford_sub_0_2 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')
    harvard_oxford_sub_25_1 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
    harvard_oxford_sub_25_2 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    harvard_oxford_sub_50_1 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm')
    harvard_oxford_sub_50_2 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
    harvard_oxford_sub_1 = datasets.fetch_atlas_harvard_oxford('sub-prob-1mm')
    harvard_oxford_sub_2 = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')

    # Multi Subject Dictionary Learning Atlas
    msdl = datasets.fetch_atlas_msdl()
    
    # Juelich Atlases
    juelich_0_1 = datasets.fetch_atlas_juelich('maxprob-thr0-1mm')  
    juelich_0_2 = datasets.fetch_atlas_juelich('maxprob-thr0-2mm')
    juelich_25_1 = datasets.fetch_atlas_juelich('maxprob-thr25-1mm') 
    juelich_25_2 = datasets.fetch_atlas_juelich('maxprob-thr25-2mm')
    juelich_50_1 = datasets.fetch_atlas_juelich('maxprob-thr50-1mm')
    juelich_50_2 = datasets.fetch_atlas_juelich('maxprob-thr50-2mm')
    juelich_1 = datasets.fetch_atlas_juelich('prob-1mm')         
    juelich_2 = datasets.fetch_atlas_juelich('prob-2mm')

    # Smith ICA Atlas and Brain Maps 2009
    smith = datasets.fetch_atlas_smith_2009()

    # ICBM tissue probability
    icbm = datasets.fetch_icbm152_2009()

    # Allen RSN networks
    allen = datasets.fetch_atlas_allen_2011()

    # Pauli subcortical atlas
    pauli_subcortex_prob = datasets.fetch_atlas_pauli_2017('prob')
    pauli_subcortex_det = datasets.fetch_atlas_pauli_2017('det')

    # Dictionaries of Functional Modes (“DiFuMo”) atlas
    difumo_64_2 = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2)
    difumo_128_2 = datasets.fetch_atlas_difumo(dimension=128, resolution_mm=2)
    difumo_256_2 = datasets.fetch_atlas_difumo(dimension=256, resolution_mm=2)
    difumo_512_2 = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=2)
    difumo_1024_2 = datasets.fetch_atlas_difumo(dimension=1024, resolution_mm=2)
     
    difumo_64_3 = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=3)
    difumo_128_3 = datasets.fetch_atlas_difumo(dimension=128, resolution_mm=3)
    difumo_256_3 = datasets.fetch_atlas_difumo(dimension=256, resolution_mm=3)
    difumo_512_3 = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=3)
    difumo_1024_3 = datasets.fetch_atlas_difumo(dimension=1024, resolution_mm=3)
    
    # AAL templates
    #aal_spm5 = datasets.fetch_atlas_aal('SPM5')
    #aal_spm8 = datasets.fetch_atlas_aal('SPM8')
    aal_spm12 = datasets.fetch_atlas_aal('SPM12')
    
    # Talairach atlas
    talairach_hemi = datasets.fetch_atlas_talairach('hemisphere')
    talairach_lobe = datasets.fetch_atlas_talairach('lobe') 
    talairach_gyrus = datasets.fetch_atlas_talairach('gyrus')
    talairach_tissue = datasets.fetch_atlas_talairach('tissue') 
    talairach_ba = datasets.fetch_atlas_talairach('ba')
    
    # Schaefer 2018 atlas
    schaefer_100_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1)
    schaefer_200_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
    schaefer_300_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=7, resolution_mm=1)
    schaefer_400_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1)
    schaefer_500_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=500, yeo_networks=7, resolution_mm=1)
    schaefer_600_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=600, yeo_networks=7, resolution_mm=1)
    schaefer_700_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=700, yeo_networks=7, resolution_mm=1)
    schaefer_800_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=1)
    schaefer_900_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=900, yeo_networks=7, resolution_mm=1)
    schaefer_1000_7_1 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1)
    
    schaefer_100_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    schaefer_200_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    schaefer_300_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=7, resolution_mm=2)
    schaefer_400_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    schaefer_500_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=500, yeo_networks=7, resolution_mm=2)
    schaefer_600_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=600, yeo_networks=7, resolution_mm=2)
    schaefer_700_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=700, yeo_networks=7, resolution_mm=2)
    schaefer_800_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=7, resolution_mm=2)
    schaefer_900_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=900, yeo_networks=7, resolution_mm=2)
    schaefer_1000_7_2 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=2)
    
    schaefer_100_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17, resolution_mm=1)
    schaefer_200_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=1)
    schaefer_300_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=17, resolution_mm=1)
    schaefer_400_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=1)
    schaefer_500_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=500, yeo_networks=17, resolution_mm=1)
    schaefer_600_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=600, yeo_networks=17, resolution_mm=1)
    schaefer_700_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=700, yeo_networks=17, resolution_mm=1)
    schaefer_800_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=17, resolution_mm=1)
    schaefer_900_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=900, yeo_networks=17, resolution_mm=1)
    schaefer_1000_17_1 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=1)
    
    schaefer_100_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17, resolution_mm=2)
    schaefer_200_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=2)
    schaefer_300_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=17, resolution_mm=2)
    schaefer_400_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    schaefer_500_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=500, yeo_networks=17, resolution_mm=2)
    schaefer_600_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=600, yeo_networks=17, resolution_mm=2)
    schaefer_700_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=700, yeo_networks=17, resolution_mm=2)
    schaefer_800_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=800, yeo_networks=17, resolution_mm=2)
    schaefer_900_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=900, yeo_networks=17, resolution_mm=2)
    schaefer_1000_17_2 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=2)

    atlas_types = {
                    'Harvard_Oxford cort 0 x 1': harvard_oxford_cort_0_1.maps,
                   'Harvard_Oxford cort 0 x 2': harvard_oxford_cort_0_2.maps,
                   'Harvard_Oxford cort 25 x 1': harvard_oxford_cort_25_1.maps,
                   'Harvard_Oxford cort 25 x 2': harvard_oxford_cort_25_2.maps,
                   'Harvard_Oxford cort 50 x 1': harvard_oxford_cort_50_1.maps,
                   'Harvard_Oxford cort 50 x 2': harvard_oxford_cort_50_2.maps,
                   'Harvard_Oxford cort 1': (harvard_oxford_cort_1.maps, harvard_oxford_cort_1.labels[1:]),
                   'Harvard_Oxford cort 2': (harvard_oxford_cort_2.maps,harvard_oxford_cort_2.labels[1:]),
                   'Harvard_Oxford cortl 0 x 1': harvard_oxford_cortl_0_1.maps,
                   'Harvard_Oxford cortl 0 x 2': harvard_oxford_cortl_0_2.maps,
                   'Harvard_Oxford cortl 25 x 1': harvard_oxford_cortl_25_1.maps,
                   'Harvard_Oxford cortl 25 x 2': harvard_oxford_cortl_25_2.maps,
                   'Harvard_Oxford cortl 50 x 1': harvard_oxford_cortl_50_1.maps,
                   'Harvard_Oxford cortl 50 x 2': harvard_oxford_cortl_50_2.maps,
                   'Harvard_Oxford cortl 1': (harvard_oxford_cortl_1.maps,harvard_oxford_cortl_1.labels[1:]),
                   'Harvard_Oxford cortl 2': (harvard_oxford_cortl_2.maps,harvard_oxford_cortl_2.labels[1:]),
                   'Harvard_Oxford sub 0 x 1': harvard_oxford_sub_0_1.maps,
                   'Harvard_Oxford sub 0 x 2': harvard_oxford_sub_0_2.maps,
                   'Harvard_Oxford sub 25 x 1': harvard_oxford_sub_25_1.maps,
                   'Harvard_Oxford sub 25 x 2': harvard_oxford_sub_25_2.maps,
                   'Harvard_Oxford sub 50 x 1': harvard_oxford_sub_50_1.maps,
                   'Harvard_Oxford sub 50 x 2': harvard_oxford_sub_50_2.maps,
                   'Harvard_Oxford sub 1': (harvard_oxford_sub_1.maps, harvard_oxford_sub_1.labels[1:]),
                   'Harvard_Oxford sub 2': (harvard_oxford_sub_2.maps, harvard_oxford_sub_2.labels[1:]),
                   
                   'Juelich 0 x 1' : juelich_0_1.maps,
                   'Juelich 0 x 2' : juelich_0_2 .maps,
                   'Juelich 25 x 1' : juelich_25_1.maps, 
                   'Juelich 25 x 2' : juelich_25_2.maps, 
                   'Juelich 50 x 1' : juelich_50_1.maps,
                   'Juelich 50 x 2' : juelich_50_2.maps, 
                   'Juelich 1' : (juelich_1.maps, juelich_1.labels[1:]),      
                   'Juelich 2' : (juelich_2.maps,juelich_2.labels[1:]),
                   
                    'MSDL': (msdl.maps, msdl.labels), 
                    
                    'Smith 10 RSNs': smith.rsn10,
                    'Smith 20 RSNs': smith.rsn20,
                    'Smith 70 RSNs': smith.rsn70,
                    'Smith 10 Brainmap': smith.bm10,
                    'Smith 20 Brainmap': smith.bm20,
                    'Smith 70 Brainmap': smith.bm70,
                    
                    'ICBM T1-Weighted': icbm['t1'],
                    'ICBM T2-Weighted': icbm['t2'],
                    'ICBM T-2 Relaxometry': icbm['t2_relax'],
                    'ICBM White Matter': icbm['wm'],
                    'ICBM Cerebrospinal' : icbm['csf'],
                    'ICBM Proton Density Weighted' : icbm['pd'],
                    'ICBM Grey Matter': icbm['gm'],
                    'ICBM Eye Mask': icbm['eye_mask'],
                    'ICBM Face Mask': icbm['face_mask'],
                    'ICBM Skull Mask': icbm['mask'],
                    'ICBM WM x GM x CSF': [icbm['wm'], icbm['gm'], icbm['csf']],
                    
                    
                    'Allen 28': (allen.rsn28, list(allen.networks)[0] + 
                                 list(allen.networks)[1] + list(allen.networks)[2] + 
                                 list(allen.networks)[3] + list(allen.networks)[4] + 
                                 list(allen.networks)[5] + list(allen.networks)[6]),
                    
                    'Allen 75': allen.maps,
                    
                    'Pauli Subcortex Prob': (pauli_subcortex_prob.maps, pauli_subcortex_prob.labels),
                    'Pauli Subcortex Det': pauli_subcortex_det.maps,
                    
                    'DiFuMo 64 x 2': (difumo_64_2.maps, [n[1] for n in list(difumo_64_2.labels)]),
                    'DiFuMo 128 x 2': (difumo_128_2.maps, [n[1] for n in list(difumo_128_2.labels)]),
                    'DiFuMo 256 x 2': (difumo_256_2.maps, [n[1] for n in list(difumo_256_2.labels)]),
                    'DiFuMo 512 x 2': (difumo_512_2.maps, [n[1] for n in list(difumo_512_2.labels)]),
                    'DiFuMo 1024 x 2': (difumo_1024_2.maps,[n[1] for n in list(difumo_1024_2.labels)]),
                    'DiFuMo 64 x 3': (difumo_64_3.maps, [n[1] for n in list(difumo_64_3.labels)]),
                    'DiFuMo 128 x 3': (difumo_128_3.maps,[n[1] for n in list(difumo_128_3.labels)]),
                    'DiFuMo 256 x 3': (difumo_256_3.maps, [n[1] for n in list(difumo_256_3.labels)]),
                    'DiFuMo 512 x 3': (difumo_512_3.maps,[n[1] for n in list(difumo_512_3.labels)]),
                    'DiFuMo 1024 x 3': (difumo_1024_3.maps,[n[1] for n in list(difumo_1024_3.labels)]),
                    
                    #'AAL SPM5' : aal_spm5.maps,
                    #'AAL SPM8' : aal_spm8.maps,
                    'AAL SPM12' : aal_spm12.maps,
                    
                    'Talairach Hemi' : talairach_hemi.maps,
                    'Talairach Lobe' : talairach_lobe.maps,
                    'Talairach Gyrus' : talairach_gyrus.maps,
                    'Talairach Tissue' : talairach_tissue.maps,
                    'Talairach Ba' : talairach_ba.maps,
                    
                    'Schaefer 100 x 7 x 1' : schaefer_100_7_1.maps,
                    'Schaefer 200 x 7 x 1' : schaefer_200_7_1.maps,
                    'Schaefer 300 x 7 x 1' : schaefer_300_7_1.maps,
                    'Schaefer 400 x 7 x 1' : schaefer_400_7_1.maps,
                    'Schaefer 500 x 7 x 1' : schaefer_500_7_1.maps,
                    'Schaefer 600 x 7 x 1' : schaefer_600_7_1.maps,
                    'Schaefer 700 x 7 x 1' : schaefer_700_7_1.maps,
                    'Schaefer 800 x 7 x 1' : schaefer_800_7_1.maps,
                    'Schaefer 900 x 7 x 1' : schaefer_900_7_1.maps,
                    'Schaefer 1000 x 7 x 1' : schaefer_1000_7_1.maps,
                    
                    'Schaefer 100 x 7 x 2' : schaefer_100_7_2.maps,
                    'Schaefer 200 x 7 x 2' : schaefer_200_7_2.maps,
                    'Schaefer 300 x 7 x 2' : schaefer_300_7_2.maps,
                    'Schaefer 400 x 7 x 2' : schaefer_400_7_2.maps,
                    'Schaefer 500 x 7 x 2' : schaefer_500_7_2.maps,
                    'Schaefer 600 x 7 x 2' : schaefer_600_7_2.maps,
                    'Schaefer 700 x 7 x 2' : schaefer_700_7_2.maps,
                    'Schaefer 800 x 7 x 2' : schaefer_800_7_2.maps,
                    'Schaefer 900 x 7 x 2' : schaefer_900_7_2.maps,
                    'Schaefer 1000 x 7 x 2' : schaefer_1000_7_2.maps,
                    
                    'Schaefer 100 x 17 x 1' : schaefer_100_17_1.maps,
                    'Schaefer 200 x 17 x 1' : schaefer_200_17_1.maps,
                    'Schaefer 300 x 17 x 1' : schaefer_300_17_1.maps,
                    'Schaefer 400 x 17 x 1' : schaefer_400_17_1.maps,
                    'Schaefer 500 x 17 x 1' : schaefer_500_17_1.maps,
                    'Schaefer 600 x 17 x 1' : schaefer_600_17_1.maps,
                    'Schaefer 700 x 17 x 1' : schaefer_700_17_1.maps,
                    'Schaefer 800 x 17 x 1' : schaefer_800_17_1.maps,
                    'Schaefer 900 x 17 x 1' : schaefer_900_17_1.maps,
                    'Schaefer 1000 x 17 x 1' : schaefer_1000_17_1.maps,
                    
                    'Schaefer 100 x 17 x 2' : schaefer_100_17_2.maps,
                    'Schaefer 200 x 17 x 2' : schaefer_200_17_2.maps,
                    'Schaefer 300 x 17 x 2' : schaefer_300_17_2.maps,
                    'Schaefer 400 x 17 x 2' : schaefer_400_17_2.maps,
                    'Schaefer 500 x 17 x 2' : schaefer_500_17_2.maps,
                    'Schaefer 600 x 17 x 2' : schaefer_600_17_2.maps,
                    'Schaefer 700 x 17 x 2' : schaefer_700_17_2.maps,
                    'Schaefer 800 x 17 x 2' : schaefer_800_17_2.maps,
                    'Schaefer 900 x 17 x 2' : schaefer_900_17_2.maps,
                    'Schaefer 1000 x 17 x 2' : schaefer_1000_17_2.maps
                    }
    
    return atlas_types

#%%
def parcellized_brain_vecs(brain_img, atlas_dict, atlas_name):
    '''
    Parameters:
        brain_img : str or list
            File name or list of file names
            
        atlas_name : str
            Name of brain parcellation technique
            
        atlas_dict : dict
            Dictionary of parcellation atlases and their maps and labels
    
    Returns: np array
       list, each value is the average brain activity intensity level of each brain region  
    '''
    
    
    # use nifti maps attribute for regional boundaries
    atlas = atlas_dict[atlas_name]
    if type(atlas) == tuple:
        atlas = atlas[0]
    masker = input_data.NiftiMapsMasker(atlas, resampling_target = 'data', 
                                        t_r = 2, detrend = True).fit()

    # create array of voxel intensity based brain parcellation 
    roi_vals = masker.transform(brain_img)
    
    
    return roi_vals.tolist()

#%%

def plot_parcellation_map(atlas_name, atlas_types, cut_coords = (0,0,0)):
    '''
    Purpose:
        Plot visualization to show the areas of the brain divivded in parcellation atlas
        
    Parameters:
        atlas_name : str
            Name of brain parcellation technique
            
        atlas_types : dict
            Dictionary of parcellation atlases and their maps and labels
            
        cut_coords : tuple
            Coordinates for brain visualization
    
    Returns: np array
        2-dim array, each value is the average brain activity intensity level of each brain region  
    '''
    
    atlas_map = atlas_types[atlas_name]
    
    if type(atlas_map) == tuple:
        atlas = atlas_map[0]
        
    else:
        atlas = atlas_map
    
    if type(atlas) == str or list:
        dim = get_data(atlas).ndim
        
    else:
        dim = atlas.ndim
    
    if dim == 4:
        plotting.plot_prob_atlas(atlas, cut_coords = cut_coords, title = atlas_name, black_bg=True)
        plt.show()
        
    else:
        plotting.plot_roi(atlas, cut_coords=cut_coords, title = atlas_name, black_bg = True)
        plt.show()
        
    
#%%

def img_to_arr(file):
    '''
    Purpose: Transform brain image into an array
    
    Parameters:
        file : str
            Brain iamge
    
    Returns: 4-D or 3-D array representative of input brain file 
    '''  
    
    array = image.get_data(file)
    
    return array

#%%
def img_to_vec(image, new_total):
    '''
    Purpose: Transform a long array into a compressed vector (compressed being the length of the vector
                                                              is smaller than the length of the array)
    
    Parameters:
        old_array : array (any dimensions)
            In this case, 3-D numpy arrays representing each brain image file
            
        new_toral : int
            Length of output vector
    
    Returns: Vector of mean values from original array (1-D list)
    '''    
   
    # intialize 
    means = []
    array = img_to_arr(image)
    
    # reshape array into 1-D
    total_nums = np.prod(array.shape)
    array = array.reshape(total_nums)
    
    # create list of values that will be used as upper and lower bounds for grouping
    length = total_nums/new_total
    mul = [x for x in range(total_nums) if x % round(length, 0)  == 0]
    
    if len(mul) != new_total:
        mul = [x for x in range(total_nums) if x % math.ceil(length)  == 0]
        if len(mul) != new_total:
            mul = [x for x in range(total_nums) if x % math.floor(length)  == 0]
            if len(mul) != new_total:
                return 'ERROR Choose a different number'
            
            
    # define grouping variable
    grouping = total_nums / new_total

    # create list of mean values of each grouped set
    
    # if input array length is divisible by new vector length
    if total_nums % new_total == 0:
        for x in mul:
            upper_x = int(x + grouping)
            mean = array[x:upper_x].mean()
            means.append(mean)
    
    #if it is not
    else:
        for x in mul[:-1]:
            upper_x = int(x + grouping)
            mean = array[x:upper_x].mean()
            means.append(mean)
        
        # add the last grouping separately as the remaining values will be left over
        last_mean = array[mul[-1]:total_nums].mean()
        means.append(last_mean)
    
    # return vector
    return means


#%%

def expand_df(df, column):
    '''
    Purpose: 
        Turn a column of vectors into multiple columns of floats 
            
        df : pandas dataframe
            Dataframe with vectors for each parcellation atlas
            
        column : str
            Specfic parcellation atlas
    
    Returns: 
        Dataframe of multiple columns
    ''' 
    
    
    new_df = pd.DataFrame()
    row_1 = list(df.index)[0]
    if type(df[column][row_1]) == list or type(df[column][row_1]) == tuple:
        nums = list(range(1, len(df[column][row_1]) +1))
        cols = [column+'_'+str(num) for num in nums]
        new_df[cols] = pd.DataFrame(df[column].tolist(), index=df.index)
        
    else:
        new_df[column] = df[column]
    
    return new_df
#%%

def standardize(prime, parcellation, df):
    '''
    Purpose: 
        Turn column of vectors into standardized array of arrays for sklearn classifiers
        
        prime : str
            Specific priming condition (congruent, incongruent, neutral)
            
        parcellation : str
            Specific parcellation atlas
            
        df : pandas dataframe
            Dataframe with vectors for each parcellation atlas
    
    Returns: 
        array of standardized vectors
    ''' 

    # condense dataframe to just rows with specific priming and parcellation
    prime_df = df[df.priming == prime]
    prime_df = prime_df[[parcellation]]
    
    # turn column of vectors to multiple columns of floats
    expanded = expand_df(prime_df, parcellation)
    
    # turn dataframe to standardized array
    return preprocessing.StandardScaler().fit(expanded).transform(expanded.astype(float))


#%%

def get_neighbors(data, test_v, k, dfunc):
    '''
    Purpose: 
        Create classification algorithm by finding the k nearest 
    neighbors among all other instances in the dataset
    
    Parameters:
        data : list
            List of tuples representing each image in the dataset
        
        test_v : list
            Vector representing an image that is being tested for nearest neighbors
        
        k : int
            Number of "neighbors" or most similar vectors the algorithm is going to store in a list
            
        dfunc : function 
            Distance function used to measure distance between vectors
    
    Returns:
        List of tuples (distance of similar vectors and their classes)
    '''
    
    # intialize
    distances = []
    neighbors = []
    
    # add distances for each row and test row to list
    for vec, class_ in data:
        dist = dfunc(test_v, vec)
        distances.append((class_, dist))
    
    # sort distances by min to max
    distances.sort(key=lambda tup: tup[1])
    
    # add K most similiar neighbors
    for i in range(k):
        neighbors.append(distances[i])
    
    return neighbors

#%%

def classifier_dict():
    '''
    Purpose: 
        Create dictionary of classifiers
    
    Returns: 
        Dictionary of classifier name (with value being the function itself)
    '''  

    dict = {'SVC' : svm.SVC(), 'NuSVC' :  svm.NuSVC(), 'LinearSVC' : svm.LinearSVC(), 
            'SGD':SGDClassifier(), 'KNN': KNeighborsClassifier(), 'KMeans': NearestCentroid(), 
            'GaussianProcess': GaussianProcessClassifier(),'GaussianNB': GaussianNB(), 
            'MultinomialNB' : MultinomialNB(), 'ComplementNB': ComplementNB(), 'BernoulliNB' : BernoulliNB(), 
             'CategoricalNB': CategoricalNB(), 'Decision Tree': DecisionTreeClassifier(),'MLP' : MLPClassifier(), 
             'PassAggr': PassiveAggressiveClassifier(), 'knn_euclidean': distance.euclidean,
             'knn_braycurtis': distance.braycurtis, 'knn_canberra' : distance.canberra, 
             'knn_chebyshev': distance.chebyshev, 'knn_cityblock': distance.cityblock, 'knn_correlation': distance.correlation, 
             'knn_cosine': distance.cosine, 'knn_cosine_similarity' : cosine_similarity, 
             'knn_jensenshannon' : distance.jensenshannon, 'knn_minkowski': distance.minkowski, 
             'knn_sqeuclidean': distance.sqeuclidean, 'knn_dice': distance.dice, 'knn_hamming': distance.hamming, 
             'knn_jaccard': distance.jaccard, 'knn_kulsinski': distance.kulsinski, 
             'knn_rogerstanimoto': distance.rogerstanimoto, 'knn_russellrao': distance.russellrao, 
              'knn_sokalmichener': distance.sokalmichener, 'knn_sokalsneath': distance.sokalsneath, 'knn_yule': distance.yule}

    return dict 

#%%

def predict_class(data, test_v, k, metric):
    '''
    Purpose: 
        Based on K nearest neighbors predict what class
        test row belongs to
    
    Parameters:
        data : list of tuples
            Each tuple is (vector, respective class)
            
        test_v : list of floats
            Vector representing specific brain image being tested for nearest neighbors
        
        k : int
            Number of "neighbors" or most similar vectors the algorithm is going to store in a list
            
        metric : function
            Distance function to calculate distance between vectors
    
    Returns:
        String (or int) representing prediction of what class the vector belongs to
    '''
 

    # find neighbors
    neighbors = get_neighbors(data, test_v, k, metric)
    
    # find most frequent class in neighbors
    output_values = [tup[0] for tup in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    
    return prediction

#%%

def get_prediction(main_df, prime, parcellation, algo, k=4, X_train=0, y_train=0, X_test=0):
    '''
    Purpose: 
        Make prediction based on algorithm 
    
    Parameters:
        main_df : pandas dataframe
            Dataframe with parcelized vectors
            
        prime : str
            Specific priming condition (congruent, incongruent, neutral)
            
        parcellation: str
            Specific parcellation atlas
            
        algo : str
            Specific algorithm
            
        k : int
            Number of nearest neighbors (if algorithm involves KNN)
            
        X_train : array
            Standardized array of training data
            
        y_train : array
            Array of classes for X_train
            
        X_test : array:
            Standardized array for test data
    
    Returns: 
        List of class predictions
    '''  
    
    algorithms = classifier_dict()
    
    if 'knn_' in algo:
        df = main_df[main_df.priming == prime]
        vecs = list(df[parcellation])
        if type(vecs[0]) != list:
            vecs = [[vec] for vec in vecs]
        classes = list(df.emotion)
        data_pairs = list(zip(vecs, classes))
        predicted = make_k_predictions(data_pairs, k, algorithms[algo])
        
    else:
        predicted = algorithms[algo].fit(X_train, y_train).predict(X_test)
        
    return predicted

#%%

def get_X_y(df, priming, parcellation, variable):
    '''
    Purpose: 
        Turn list of vectors into arrays of train and testing data
    
    Parameters:
        df : pandas dataframe
            Dataframe with parcelized vectors
            
        priming : str
            Specific priming condition (congruent, incongruent, neutral)
            
        parcellation: str
            Specific parcellation atlas
            
        variable : str
            Specific variable user wants returned (X_train, X_test, y_train, y_test)
    
    Returns: 
        List/array representing train/test data/classes
    '''  
    
    X_train = list(df[(df.priming == priming) & (df.parcellation == parcellation)].X_train)[0]
    X_test = list(df[(df.priming == priming) & (df.parcellation == parcellation)].X_test)[0]
    y_train = list(df[(df.priming == priming) & (df.parcellation == parcellation)].y_train)[0]
    y_test = list(df[(df.priming == priming) & (df.parcellation == parcellation)].y_test)[0]
    
    
    if variable == 'X_train':
        return X_train
    
    if variable == 'X_test':
        return X_test
    
    if variable == 'y_train':
        return y_train
    
    if variable == 'y_test':
        return y_test

#%%

def get_actual(df, standardized_df, priming, parcellation, algorithm):
    '''
    Purpose: 
        Get list of actual classes for dataset
    
    Parameters:
        df : pandas dataframe
            Dataframe with  list of classes for vectors
            
        standardized_df : pandas dataframe
            Dataframe with  list of classes for standardized arrays
            
        priming : str
            Specific priming condition (congruent, incongruent, neutral)
            
        parcellation: str
            Specific parcellation atlas
            
        algorithm: str
            Specific algorithm
    
    Returns: 
        List of classes for corresponding vectors/arrays
    '''  
    
    # find classes for vectors 
    if 'knn_' in algorithm:
        actual = list(df[df.priming == priming].emotion)
   
        # find classes for standardized arrays
    else:
        actual = list(standardized_df[(standardized_df.priming == priming) & (standardized_df.parcellation == parcellation)].y_test)[0]
        
    return actual

#%%

def rate_classifier(predicted, actual, score_type):
    '''
    Purpose: 
        Calculate accuracy and f1 score for overall performance and performance of each class
        to analyze the performance of the classifier for specific k value
    
    Parameters:
        predicted : list
            List of class predictions for each vector 
            
        actual : list
            List of the actual classes for each vector
            
        score_type: str
            What score user wants to calculate
    
    Returns: 
        Accuracy score or F-1 score (depends on what was specified)
    '''
    
    
    
    
    if score_type == 'f1':
        score = metrics.f1_score(actual, predicted, average='weighted')
        
    elif score_type == 'anger_f1':
        score = list(metrics.f1_score(actual, predicted, average=None))[0]
        
    elif score_type == 'disgust_f1':
        score = list(metrics.f1_score(actual, predicted, average=None))[1]
        
    elif score_type == 'fear_f1':
        score = list(metrics.f1_score(actual, predicted, average=None))[2]
    
    elif score_type == 'accuracy':
        score = metrics.accuracy_score(actual, predicted)
        
    elif score_type == 'anger_accuracy':
        score = list(metrics.recall_score(actual, predicted, average=None))[0]
        
    elif score_type == 'disgust_accuracy':
        score = list(metrics.recall_score(actual, predicted, average=None))[1]
        
    elif score_type == 'fear_accuracy':
        score = list(metrics.recall_score(actual, predicted, average=None))[2]
    
    return score



#%%
def make_k_predictions(data, k, metric):
    '''
    Purpose: 
        Make predictions for K Nearest Neighbors for all data points
    
    Parameters:
        data : list of tuples
            Each tuple is (vector, respective class)
        
        k : int
            Number of nearest neighbors to calculate for
            
        metric : function
            Distance function to calculate distance between vectors
    
    Returns:
        List of predicted classes for each vector in data
    '''

    
    # intialize
    predicted = []
    
    # add predicted values for each row in data to list
    for vec, class_ in data:
        p = predict_class(data, vec, k, metric)
        predicted.append(p)
        
    return predicted

#%%

def plot_correlation(atlas_dict, data_df, parcellation, atlas_name, cmap = 'coolwarm'):
    '''
    Purpose: 
        Plots correlation map=trix between each area outlined in parcellation atlas
        
    Parameters:
        atlas_dict : dict
            Dictionary of each parcellation as keys and respective maps and labels as values
        
        data_df : pandas dataframe
            Dataframe of brain images and parcelized vectors for each parcellation atlas
            
        parcellation : str
            Column name in data_df representative of a parcellation atlas
            
        atlas_name : str
            Key name in atlas_dict represenetative of a parcellation atlas
    
    Returns:
        None, renders a correlation matrix
        showing activity of diff brain regions during emotion task
    '''
    

    # create 2D array showing ROIs for a specific atlas
    array = np.array(list(data_df[parcellation]))
    
    # create correlation matrix
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([array])[0]


    # find label for specific atlas and plot matrix
    label = atlas_dict[atlas_name][1]
    plotting.plot_matrix(correlation_matrix, title = atlas_name, labels=label, figure=(15,15), cmap=cmap)


    
