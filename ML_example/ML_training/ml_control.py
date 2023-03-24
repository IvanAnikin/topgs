
import datetime

from ML_training import Train


subname = "trained_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # _%H:%M:%S
trainer = Train.Trainer(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1, 0], visualisation_type = [1, 0, 1, 1, 0, 1, 0, 1, 0], dim=[[0, 0, 1],[1, 1, 1, 1, 1]], subname=subname, datasets_directory = "C:/ML_car/Datasets/Preprocessed/Trained",
                        model_type="DQN", video_source="Ip Esp32 Stream", object='helmet', object_threshold = 0.8, type='detective', MIN_MATCH_COUNT=10) #explorer

trainer.train(visualise=True)