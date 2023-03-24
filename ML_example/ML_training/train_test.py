
from ML_training import Train


#trainer = Train.Trainer(model_type="DQN_2", batch_size = 32, datasets_directory="C:/ML_car/Datasets/Preprocessed/fsebcardom_996df78", models_directory="C:/ML_car/Models",
#                        load_model=False, detect_objects=True, preprocess=False, dim=[[0, 0, 1],[0, 0, 0, 0, 1, 1]],  filters=(32, 64, 128), filters_dropouts=[0.2, 0.2, 0.2, 0.2],
#                        layers_after_filters=(64, 32, 16), dropouts=[0.2, 0.2, 0.2, 0.2], last_layer_activation="linear", trained_testing=False, type="explorer", random_action=False)
#
#trainer.simulate_on_dataset(file_name="f_s_e_b_c_a_r_d_o_m_combined.npy", visualise=True)


trainer = Train.Trainer(model_type="DQN_2", batch_size = 32, datasets_directory="C:/ML_car/Datasets/Preprocessed/fsebcardom_996df78", models_directory="C:/ML_car/Models",
                        load_model=False, detect_objects=True, preprocess=True, dim=[[0, 0, 1],[1, 1, 1, 1, 1, 1]],  filters=(32, 64, 128), filters_dropouts=[0.2, 0.2, 0.2, 0.2],
                        layers_after_filters=(64, 32, 16), dropouts=[0.2, 0.2, 0.2, 0.2], last_layer_activation="linear", trained_testing=False, type="explorer", random_action=False,
                        dataset_type=[1, 0, 0, 0, 0, 1, 1, 1, 0, 0],visualisation_type = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

trainer.trained_control(visualise=True)
