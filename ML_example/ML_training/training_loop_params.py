
import gc
import numpy as np
import os
from operator import itemgetter
import matplotlib.pyplot as plt

from ML_training import Train
from ML_training import params


train = False
random_action=True

rewards_development = False
rewards_development_steps = 20
model_path = "Trained" #"Training" #"Trained"

trained_testing = True
non_deletable_params = ['gc', 'Train', 'params', 'train', 'trained_testing', 'non_deletable_params']
average_rewards = []
count = 0
dataset_name_full = "C:/ML_car/Models/same_action/Training/DQN_2/[[0, 0, 1], [0, 0, 0, 0, 1, 1]]/(32, 64, 128)_[0.2, 0.2, 0.2, 0.2]_(64, 32, 16)_[0.2, 0.2, 0.2, 0.2]_linear___random.npy"
dataset_name = "random"
dataset= np.load(dataset_name_full, allow_pickle=True)
average_action = np.average(dataset[:,1])
average_reward = np.average(dataset[:,2])
#average_rewards.append([str(count), np.round(average_reward, 3), dataset.shape[0], average_action, dataset_name])
#ount+=1

for step in params.params:
    dim = step[0]
    filters = step[1]
    filters_dropouts = step[2]
    layers_after_filters = step[3]
    dropouts = step[4]
    last_layer_activation = step[5]

    if(train):
        trainer = Train.Trainer(model_type="DQN_2", batch_size=32, trained_testing=trained_testing, type="explorer",
                                datasets_directory="C:/ML_car/Datasets/Preprocessed/fsebcardom_996df78",models_directory="C:/ML_car/Models",load_model=True, detect_objects=True,
                                preprocess=False, dim=dim, filters=filters,filters_dropouts=filters_dropouts,layers_after_filters=layers_after_filters, dropouts=dropouts,
                                last_layer_activation=last_layer_activation, random_action=random_action)

        trainer.simulate_on_dataset(file_name="f_s_e_b_c_a_r_d_o_m_combined.npy", visualise=True)

        for name in dir():
            if not name.startswith('_') and name not in non_deletable_params:
                del globals()[name]
        gc.collect()

    else:
        model_subname = str(dim)
        model_subname2 = "{filters}_{filters_dropouts}_{layers_after_filters}_{dropouts}_{last_layer_activation}".format(filters=str(filters),
                            filters_dropouts=str(filters_dropouts), layers_after_filters=str(layers_after_filters), dropouts=str(dropouts), last_layer_activation=last_layer_activation)
        dataset_name_full = "C:/ML_car/Models/same_action/{model_path}/DQN_2/{subname}/{subname2}.npy".format(model_path=model_path, subname=model_subname, subname2=model_subname2)
        dataset_name = "{subname}_|_{subname2}".format(subname=model_subname, subname2=model_subname2)
        if os.path.exists(dataset_name_full):
            dataset = np.load(dataset_name_full, allow_pickle=True)

            if(rewards_development):
                total_len = len(dataset)
                average_rewards_by_step = []
                for i in range(rewards_development_steps):
                    step_reward_total = 0
                    rewards = 0
                    for row in dataset:
                        timestep, action, reward = row
                        if(timestep<i*total_len/rewards_development_steps):
                            step_reward_total += reward
                            rewards += 1
                    if(rewards!=0): average_rewards_by_step.append((step_reward_total)/rewards)
                print(average_rewards_by_step)
                print(np.round(np.average(average_rewards_by_step),2))
            else:
                average_action = np.average(dataset[:,1])
                average_reward = np.average(dataset[:,2])
                average_rewards.append([str(count), np.round(average_reward, 3), dataset.shape[0], average_action, dataset_name])
                count+=1

if(not train and not rewards_development):
    average_rewards=sorted(average_rewards,key=itemgetter(1))
    print(*average_rewards, sep='\n')
    average_rewards=np.array(average_rewards)
    names = average_rewards[:,0]
    rewards = average_rewards[:,1]

    y_pos = np.arange(len(names))

    plt.bar(y_pos, rewards, align='center', alpha=0.5)
    plt.xticks(y_pos, names)
    plt.ylabel('Avg. reward')
    plt.title('Achievements dependency on params')

    plt.show()

    print()
    print()
    for dataset in average_rewards:
        print("{count}: {name}".format(count=dataset[0], name=dataset[4]))
