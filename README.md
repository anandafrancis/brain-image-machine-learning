# Machine Learning with fMRI NeuroImages 

My project analyzes a dataset obtained from Northeastern University’s Affective and Brain Sciences Lab, in which thirty participants in a functional Magnetic Resonance Imaging (fMRI) scanner were shown images that elicit anger, fear, or disgust after being primed with congruent, incongruent, or neutral emotion words. The brain data files received are preprocessed three-dimensional beta maps, where the value at each voxel is representative of the participant’s level of brain activity during this task. 

I used this data to answer the following research questions:
    Which emotion state (anger, fear, disgust) can be most accurately predicted across the different priming conditions, parcellation techniques and classifiers ?
    How does the priming condition affect the accuracy of our classifiers?
    What classifier will have the highest prediction accuracy?
    How does parcellation technique affect the accuracy of our classifiers?
 
 
 ## Files
* `functions.py`: python file with all the functions used in .ipynb file
* `ml_neuro_research_report.ipynb`: jupyter notebook file with research write-up, dataframes and visualizations
* `ml_neuro_dataframes.ipynb`: jupyter notebook file of code used to create brain_data.csv, train_test_data.csv, testing_data.csv
* `brain.data.csv.zip`: compressed file of all the brain images as vectors for each parcellation technique
* `train_test_data.csv`: file of pandas dataframe of standardized arrays of the parcellation vectors 
* `testing_data.csv`: file of pandas dataframe of accuracy and f-1 scores for each priming condition x parcellation technique x algorithm combination
* `data`: file folder of 270 nii.gz fMRI images 
