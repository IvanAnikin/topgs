
import numpy as np
import os
from Managers import dataset_manager
from Managers.preporcess_manager import PreprocessManager

#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1, 0], subname="trained_2021-09-07_19-57-37", datasets_directory="C:/ML_car/Datasets/Trained")
#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1], subname="2021-06-15_08-13-43", datasets_directory="C:/ML_car/Datasets/Preprocessed/673243b") # video from 260/280 frome
#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1], subname="new_03.06_2", datasets_directory="C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets/Dist_normalisation")
datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], subname="", datasets_directory="C:/ML_car/Datasets/Preprocessed/fsebcardom_996df78") # video from 260/280 frome
#datasetManager.dataset_name_full="C:/ML_car/Datasets/Preprocessed/fsebcardom_996df78/fsebcardomf_s_e_b_c_a_r_d_o.npy"
#datasetManager.combine_o_m()
#datasetManager.visualise_dataset()
#datasetManager.visualise_datasets_numbers(name_base="f_s_e_b_c_a_r_d_o")
#datasetManager.preprocess_datasets(new_directory=datasetManager.datasets_directory+"/673243b/", name_base="f_s_e_b_c_a_r_d_o", visualise=True)
#datasetManager.add_monodepth(testing=True)

#datasetManager.update_rewards()
datasetManager.combine_all_datasets(name="f_s_e_b_c_a_r_d_o_m_combined")