import time
import datetime
import socket
import cv2
import numpy as np
import sys
from bidict import bidict
import time
from termcolor import colored
import os

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import ML_training.Agent as Agents
from Managers import dataset_manager
from Managers import cv_manager

class Trainer():

    def __init__(self,dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1],visualisation_type = [1, 1, 1, 1, 1, 1, 1, 1, 1],subname = "new",datasets_directory = "C:/ML_car/Datasets/Preprocessed/673243b",
        vid_dim=[120, 160], actions = [0, 2, 3], model_subname = "", model_subname2 = "", load_model=True, total_len=0,
        type="explorer", model_type="Actor_Critic", video_source = "Ip Esp32 Stream",
        minimal_distance = 20, object = 'helmet', object_threshold = 0.8, MIN_MATCH_COUNT = 7,
        models_directory="C:/ML_car/Models",images_directory = '../src/img/', objects_model_path="C:/Users/ivana/PycharmProjects/Machine_Learning/ComputerVision/AutopilotCar/Managers/CV_Models",
        plot_model_image = False, detect_objects = True, preprocess = False, scale_percent = 50, actions_stats_dir="C:/ML_car/Models/same_action",
        dim=[[0, 0, 1],[1, 1, 1, 1, 1, 1]], optimizer = Adam(learning_rate=0.01), batch_size = 32, trained_testing=False, random_action=False,
        filters=(16, 32, 64), filters_dropouts=[0.2, 0.2, 0.2, 0.2], dropouts=[0.2, 0.2, 0.2], last_layer_activation="linear", layers_after_filters=(64, 32, 16),
        dataset_vars=["last_frame", "resized", "canny_edges", "blackAndWhite", "contours", "action", "reward", "distance", "objects", "monodepth"]
    ):
        super().__init__()

        if model_subname=="": model_subname = str(dim)
        if model_subname2=="": model_subname2 = "{filters}_{filters_dropouts}_{layers_after_filters}_{dropouts}_{last_layer_activation}".format(filters=str(filters),
                            filters_dropouts=str(filters_dropouts), layers_after_filters=str(layers_after_filters), dropouts=str(dropouts), last_layer_activation=last_layer_activation)
        if random_action: model_subname2+="___random"

        self.video_source = video_source    #"Webcam"
        self.host = "192.168.1.159"  # ESP32 IP in local network
        self.port = 80  # ESP32 Server Port
        self.stream_port = 81
        self.dataset_type = dataset_type
        self.minimal_distance = minimal_distance
        self.object = object
        self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
        self.trained_testing = trained_testing
        self.random_action = random_action

        self.detect_objects = detect_objects
        self.actions_stats_dir = actions_stats_dir
        self.preprocess = preprocess
        self.actions = actions
        self.model_type=model_type
        self.type = type
        self.model_subname = model_subname
        self.model_subname2 = model_subname2
        self.total_timesteps=0
        self.batch_size = batch_size
        self.avg_timestep=0
        self.actions_relations = bidict({0:0, 1:2, 2:3})    # model:dataset
        self.default_y = 0
        self.output_size = len(actions)
        self.dim = dim[0]
        self.vid_inputs = dim[1]
        self.vid_dim = np.array(np.append(np.array(vid_dim).astype(int), sum(1 for e in self.vid_inputs if e is not 0))).astype(int)
        self.state_size = [self.dim, self.vid_dim]
        self.process_string = '\r Timestep: {timestep}/{dataset_len} - {percentage}% | total:{total_timesteps}/{total_len} - {total_percentage}% | model_actions avg: {model_actions_avg} | dataset time: {total_time} | ' \
                              'average step time: {step_time} | remaining time: dataset-{time_left} total-{total_time_left}'
        self.trained_process_string = '\r Timestep: {timestep} | total:{total_timesteps} | model_actions avg: {model_actions_avg} | total time: {total_time} | distance: {distance} | action: {action}'

        self.model_same_action = []

        #model_subname += "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        q_name = "DQN_qnetwork" #+ "_" + model_subname
        t_name = "DQN_target" #+ "_" + model_subname

        self.datasetManager = dataset_manager.DatasetManager(dataset_type = dataset_type, subname=subname,datasets_directory=datasets_directory, type=type, dim=dim,
                                                             objects_model_path=objects_model_path, scale_percent=scale_percent, visualisation_type=visualisation_type,
                                                             dataset_vars=dataset_vars)

        self.total_len = total_len

        if model_type=="DQN": self.Agent = Agents.DQN(state_size = self.state_size, actions=actions, optimizer=optimizer, models_directory=models_directory,
                                load_model=load_model, models_names=bidict({"q_name":q_name, "t_name":t_name}), model_subname=self.model_subname)
        elif model_type=="Actor_Critic": self.Agent = Agents.Actor_Critic(state_size = self.state_size, actions=actions, optimizer=optimizer,
                                models_directory=models_directory, load_model=load_model, model_name="Actor_critic", model_subname=model_subname, minimal_distance=minimal_distance)
        elif model_type=="DQN_2": self.Agent = Agents.DQN_2(state_size = self.state_size, actions=actions, optimizer=optimizer, models_directory=models_directory,
                                load_model=load_model, models_names=bidict({"q_name":q_name, "t_name":t_name}), model_subname=self.model_subname, plot_model_image=plot_model_image,
                                model_subname2 = model_subname2, filters=filters, dropouts=dropouts, filters_dropouts=filters_dropouts, last_layer_activation=last_layer_activation,
                                layers_after_filters=layers_after_filters, random_action=self.random_action)
        self.Agent.visualise_model()

        self.cv_manager = cv_manager.CV_Manager(threshold=object_threshold, name = object, images_directory=images_directory)

    def trained_control(self, visualise=False):

        model_actions = []
        dataset=[]
        start = time.time()

        sock = socket.socket()
        sock.connect((self.host, self.port))
        print("Connecting to " + self.host + ":" + str(self.port) + " - OK")

        timestep = 0

        while True:

            if (self.video_source == "Webcam"): cap = cv2.VideoCapture(0)
            if (self.video_source == "Ip Esp32 Stream"): cap = cv2.VideoCapture(
                'http://' + self.host + ':' + str(self.stream_port) + '/stream')

            ok, frame = cap.read()
            if (not ("action" in locals())): last_frame = frame.copy()
            cap.release()

            try:
                received = sock.recv(1024).decode("utf-8")  # /512

                if (len(received) < 6):
                    distance = received.strip('\r\n')  # line, buffer = received.split('\n', 1)

                    if (distance != ''):

                        if(self.type=='detective'):
                            #object_frame, object_found = self.cv_manager.find_object_2(frame)
                            object_frame, object_found = self.cv_manager.find_object(frame, MIN_MATCH_COUNT=self.MIN_MATCH_COUNT, count=0)
                            if(object_found):
                                cv2.imshow('OBJECT FOUND', object_frame)
                                k = cv2.waitKey(5000) & 0xff
                                if k == 27:
                                    break
                                print()
                                print(colored("Object '{object}' found on timestep '{timestep}'".format(object=self.object, timestep=timestep), 'green'))
                                cv2.imwrite('C:/ML_car/Found_Objects/found_{object}_{dataset_name}_{timestep}.jpg'.format(object=self.object, dataset_name=self.datasetManager.dataset_name, timestep=timestep), frame)

                        distance = int(distance)
                        state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

                        #resized,dim = self.datasetManager.preprocessManager.resized(frame=frame)
                        #canny_edges = state[1][1][3]
                        #blackAndWhite = state[1][1][4]
                        #monodepth = state[1][5]
                        # Detect objects and calculate reward
                        objects = []
                        if (self.type == "explorer" or self.type == "detective") and self.detect_objects:
                            if (self.preprocess):
                                objects = self.datasetManager.preprocessManager.objects_detection(frame=frame)
                            else:
                                objects = row[8]
                        reward = 0
                        if ("last_frame" in locals()): reward = self.datasetManager.preprocessManager.reward_calculator(
                            frame=frame, last_frame=last_frame, distance=distance, objects=objects)

                        # Saving the process
                        row = []
                        if ("action" in locals()):
                            count = 0
                            for dataset_type_row in self.dataset_type:
                                if (dataset_type_row):
                                    row.append(locals()[self.datasetManager.dataset_vars[count]])
                                else:
                                    row.append(None)
                                count += 1
                            dataset.append(row)
                            self.datasetManager.save_data(np.array(dataset))

                            visualised_row = row
                            if (self.vid_inputs[3]):
                                visualised_row[2] = state[1][0][3]
                            if (self.vid_inputs[4]):
                                visualised_row[3] = state[1][0][4]
                            if (visualise):
                                self.datasetManager.visualise_row(row=visualised_row)
                                k = cv2.waitKey(500) & 0xff
                                if k == 27:
                                    break

                        # Calculating action
                        if self.model_type=="DQN":
                            action = self.Agent.act(state=state)
                            if(distance<self.minimal_distance):
                                action = 2 #left
                        elif self.model_type=="Actor_Critic":
                            action_probs, critic_value = self.Agent.act(state, distance=distance)
                            propabilities = self.convert_propabilitites(action_probs=np.array(np.squeeze(action_probs)))
                            action = np.random.choice(self.actions, p=propabilities)
                        elif(self.model_type == "DQN_2"):
                            action, current_q_value = self.Agent.act(state, training = True)
                        model_actions.append(action)

                        # Take action
                        sock.send(b"M:" + str(action).encode())  # send aciton to car

                        # Visualisation
                        if visualise:
                            sys.stdout.write(self.trained_process_string.format(
                                timestep=timestep, model_actions_avg=round(np.average(model_actions), 2), total_timesteps=self.total_timesteps,
                                total_time=time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                                distance=distance, action=action))
                            sys.stdout.flush()
                            if (timestep % 50 == 0): print()

                        last_frame = frame
                        timestep += 1
                        self.total_timesteps += 1

            except socket.error:
                pass

    def train(self, visualise=False):

        model_actions = []
        dataset=[]
        start = time.time()

        sock = socket.socket()
        sock.connect((self.host, self.port))
        print("Connecting to " + self.host + ":" + str(self.port) + " - OK")

        timestep = 0

        with tf.GradientTape(persistent=True) as tape:
            while True:

                if (self.video_source == "Webcam"): cap = cv2.VideoCapture(0)
                if (self.video_source == "Ip Esp32 Stream"): cap = cv2.VideoCapture(
                    'http://' + self.host + ':' + str(self.stream_port) + '/stream')

                ok, frame = cap.read()
                if (not ("action" in locals())): last_frame = frame.copy()
                cap.release()

                try:
                    received = sock.recv(1024).decode("utf-8")  # /512

                    if (len(received) < 6):
                        distance = received.strip('\r\n')  # line, buffer = received.split('\n', 1)

                        if (distance != ''):

                            if (self.type == 'detective'):
                                # object_frame, object_found = self.cv_manager.find_object_2(frame)
                                object_frame, object_found = self.cv_manager.find_object(frame, MIN_MATCH_COUNT=self.MIN_MATCH_COUNT,
                                                                                         count=0)
                                if (object_found):
                                    cv2.imshow('OBJECT FOUND', object_frame)
                                    k = cv2.waitKey(5000) & 0xff
                                    if k == 27:
                                        break
                                    print()
                                    print(colored("Object '{object}' found on timestep '{timestep}'".format(object=self.object, timestep=timestep), 'green'))
                                    cv2.imwrite(
                                        'C:/ML_car/Found_Objects/found_{object}_{dataset_name}_{timestep}.jpg'.format(
                                            object=self.object, dataset_name=self.datasetManager.dataset_name,
                                            timestep=timestep), frame)

                            distance = int(distance)

                            state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

                            # Calculating action
                            if self.model_type=="DQN":
                                if (distance < self.minimal_distance):
                                    action = 2  # left
                                else:
                                    action = self.Agent.act(state)
                                    #if(action==1): action=2
                                    #action = self.actions_relations[action]
                            elif self.model_type=="Actor_Critic":
                                action_probs, critic_value = self.Agent.act(state)
                                propabilities = self.convert_propabilitites(action_probs=np.array(np.squeeze(action_probs)))
                                action = np.random.choice(self.actions, p=propabilities)
                            model_actions.append(action)

                            # Take action
                            sock.send(b"M:" + str(action).encode())  # send aciton to car                                                              # -- ***

                            # Receive next state                                                            # -- ***
                            # frame =
                            # distance =
                            # Preprocess nextstate
                            next_state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

                            # Detect objects and calculate reward
                            objects=[]
                            if self.type == "explorer" and self.detect_objects: objects = self.datasetManager.preprocessManager.objects_detection(frame=frame)
                            reward=0
                            if ("last_frame" in locals()): reward = self.datasetManager.preprocessManager.reward_calculator(frame=frame, last_frame=last_frame,
                                                                                                                            distance=distance, objects=objects, object_found=object_found)

                            #print(" | " + str(action) + " | " + str(self.actions.index(action)))
                            # Storing
                            if self.model_type=="DQN": self.Agent.store(state, action, reward, next_state)
                            elif self.model_type=="Actor_Critic": self.Agent.store(critic_value, action_probs, self.actions.index(action), reward)

                            state = next_state

                            if self.model_type == "DQN" and len(self.Agent.expirience_replay) > self.batch_size: self.Agent.retrain(self.batch_size)
                            elif self.model_type == "Actor_Critic" and len(self.Agent.rewards_history) > 0: self.Agent.retrain(tape=tape)

                            # Aligning target model if on second last state
                            #if (timestep == len(dataset) - 2 and self.model_type=="DQN"): self.Agent.alighn_target_model()
                            if (timestep % 10 == 0): self.Agent.save_model(subname=self.model_subname)

                            # Saving the process
                            row = []
                            if ("action" in locals()):
                                count = 0
                                for dataset_type_row in self.dataset_type:
                                    if (dataset_type_row):
                                        row.append(locals()[self.datasetManager.dataset_vars[count]])  # row[count] = globals()[datasetManager.dataset_vars[count]]
                                    else:
                                        row.append(None)
                                    count += 1
                                dataset.append(row)
                                self.datasetManager.save_data(np.array(dataset))

                                visualised_row=row
                                if (self.vid_inputs[3]):
                                    visualised_row[2] = state[1][0][3]
                                if(self.vid_inputs[4]):
                                    visualised_row[3] = state[1][0][4]
                                if (visualise):
                                    self.datasetManager.visualise_row(row=visualised_row)
                                    k = cv2.waitKey(500) & 0xff
                                    if k == 27:
                                        break

                            # Visualisation
                            if visualise:
                                sys.stdout.write(self.trained_process_string.format(
                                    timestep=timestep, model_actions_avg=round(np.average(model_actions), 2), total_timesteps=self.total_timesteps,
                                    total_time=time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                                    distance=distance, action=action))
                                sys.stdout.flush()
                                if (timestep % 50 == 0): print()

                            last_frame = frame
                            timestep += 1
                            self.total_timesteps += 1

                except socket.error:
                    pass

    def simulate_on_dataset_2(self, file_name="", visualise=False):

        if file_name == "":dataset = self.datasetManager.dataset
        else:dataset = np.load(self.datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)

        model_actions = []
        start = time.time()

        frame = dataset[0][0]
        distance = dataset[0][7]
        state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

        with tf.GradientTape(persistent=True) as tape:
            for timestep in range(1, len(dataset)-1):                                           # -- ***

                # Calculating action
                if self.model_type=="DQN": action = self.Agent.act(state)
                elif self.model_type=="Actor_Critic":
                    action_probs, critic_value = self.Agent.act(state)
                    #print("\naction_probs: {action_probs} | critic_value: {critic_value}".format(action_probs=action_probs, critic_value=critic_value))
                    propabilities = self.convert_propabilitites(action_probs=np.array(np.squeeze(action_probs)))
                    print(action_probs)
                    action = np.random.choice(self.actions, p=propabilities)
                    #print("propabilities: {propabilities} | action: {action}".format(propabilities=propabilities, action=action))
                model_actions.append(action)
                # Take action
                # -                                                                             # -- ***

                # Receive next state                                                            # -- ***
                row = dataset[timestep]
                frame = row[0]
                distance = row[7]

                # Preprocess nextstate
                next_state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

                # Detect objects and calculate reward
                objects=[]
                if self.type == "explorer" and self.detect_objects: objects = self.datasetManager.preprocessManager.objects_detection(frame=frame)
                reward=0
                if ("last_frame" in locals()): reward = self.datasetManager.preprocessManager.reward_calculator(frame=frame, last_frame=last_frame,
                                                                                                                distance=distance, objects=objects)

                print(" | " + str(action) + " | " + str(self.actions.index(action)))
                # Storing
                if self.model_type=="DQN": self.Agent.store(state, action, reward, next_state)
                elif self.model_type=="Actor_Critic": self.Agent.store(critic_value, action_probs, self.actions.index(action), reward)

                state = next_state

                if self.model_type == "DQN" and len(self.Agent.expirience_replay) > self.batch_size: self.Agent.retrain(self.batch_size)
                elif self.model_type == "Actor_Critic" and len(self.Agent.rewards_history) > 0: self.Agent.retrain(tape=tape)

                # Aligning target model if on second last state
                if (timestep == len(dataset) - 2 and self.model_type=="DQN"): self.Agent.alighn_target_model()
                if (timestep % 10 == 0): self.Agent.save_model(subname=self.model_subname)

                # Visualisation
                if visualise:
                    sys.stdout.write(self.process_string.format(
                        timestep=timestep, dataset_len=len(dataset), percentage=round(timestep / len(dataset) * 100, 2),
                        model_actions_avg=round(np.average(model_actions), 2),
                        total_time=time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                        step_time=round((time.time() - start) / timestep, 2), time_left=time.strftime('%H:%M:%S',time.gmtime((time.time() - start) / timestep * (len(dataset) - timestep))),
                        total_timesteps=self.total_timesteps, total_len=self.total_len,
                        total_percentage=round((self.total_timesteps) / self.total_len * 100, 2),
                        total_time_left=time.strftime('%H:%M:%S', time.gmtime(
                            (time.time() - start) / timestep * (self.total_len - self.total_timesteps)))))
                    sys.stdout.flush()
                    if (timestep % 50 == 0): print()

                last_frame = frame
                self.total_timesteps += 1

    def simulate_on_dataset(self, file_name="", visualise=False):

        if file_name=="": dataset=self.datasetManager.dataset
        else: dataset = np.load(self.datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)
        if(self.total_len == 0):
            self.total_len += len(dataset)

        model_actions = []
        start = time.time()

        # Observing state of the environment
        state, action, reward = self.get_state_row_data(row=dataset[0])
        #if(self.preprocess):
        frame = dataset[0][0]
        distance = dataset[0][7]
        state2 = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

        with tf.GradientTape(persistent=True) as tape:
            for timestep in range(1, len(dataset)-1):

                # Calculating action
                if self.model_type == "DQN":
                    model_action = self.Agent.act(state, training = True)
                elif self.model_type == "Actor_Critic":
                    action_probs, critic_value = self.Agent.act(state)
                    propabilities = self.convert_propabilitites(action_probs=np.array(np.squeeze(action_probs)))
                    model_action = np.random.choice(self.actions, p=propabilities)
                if(self.model_type == "DQN_2"):
                    model_action, current_q_value = self.Agent.act(state, training = True)

                model_actions.append(model_action)

                # Observing state after taking action
                next_state, next_action, reward = self.get_state_row_data(row=dataset[timestep+1])

                # Receive next state                                                            # -- ***
                row = dataset[timestep]
                frame = row[0]
                distance = row[7]
                # Preprocess nextstate
                if (self.preprocess): next_state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

                # Detect objects and calculate reward
                objects = []
                if (self.type == "explorer" or self.type=="detective") and self.detect_objects:
                    if(self.preprocess): objects = self.datasetManager.preprocessManager.objects_detection(frame=frame)
                    else: objects=row[8]
                reward = 0
                if ("last_frame" in locals()): reward = self.datasetManager.preprocessManager.reward_calculator(
                    frame=frame, last_frame=last_frame,distance=distance, objects=objects)

                if(action == model_action): self.model_same_action.append([timestep, action, reward])

                if not self.trained_testing:
                    # Storing
                    if self.model_type == "DQN": self.Agent.store(state, action, reward, next_state)
                    elif self.model_type == "Actor_Critic": self.Agent.store(critic_value, action_probs, action, reward)

                    if self.model_type == "DQN" and len(self.Agent.expirience_replay) > self.batch_size:
                        self.Agent.retrain(self.batch_size)
                    if self.model_type == "Actor_Critic" and len(self.Agent.rewards_history) > 0:#self.batch_size
                        self.Agent.retrain(tape=tape)
                    if self.model_type == "DQN_2" and len(self.Agent.expirience_replay) > self.batch_size:
                        self.Agent.train(next_state=next_state, current_q_value=current_q_value, reward=reward, tape=tape)

                    # Aligning target model if on second last state
                    if(timestep==len(dataset)-2 and self.model_type=="DQN"): self.Agent.alighn_target_model()

                if(timestep%10==0):
                    if not self.trained_testing: self.Agent.save_model()
                    if(self.model_type=="DQN_2"): self.save_actions_stats()

                # Visualisation
                if visualise:
                    sys.stdout.write(self.process_string.format(
                        timestep=timestep, dataset_len=len(dataset), percentage=round(timestep/len(dataset)*100, 2),
                        model_actions_avg=round(np.average(model_actions),2), total_time=time.strftime('%H:%M:%S', time.gmtime(time.time()-start)),
                        step_time=round((time.time()-start)/timestep,2), time_left=time.strftime('%H:%M:%S', time.gmtime((time.time()-start)/timestep*(len(dataset)-timestep))),
                        total_timesteps=self.total_timesteps, total_len=self.total_len, total_percentage=round((self.total_timesteps)/self.total_len*100, 2),
                        total_time_left=time.strftime('%H:%M:%S', time.gmtime((time.time()-start)/timestep*(self.total_len-self.total_timesteps)))))
                    sys.stdout.flush()
                    if(timestep%50==0): print()

                #timestep+=1
                last_frame = frame
                state = next_state
                action = next_action
                self.total_timesteps += 1

    def simulate_on_datasets(self, visualise = False, type = 1):
        files = [i for i in os.listdir(self.datasetManager.datasets_directory) if
                 os.path.isfile(os.path.join(self.datasetManager.datasets_directory, i)) and self.datasetManager.dataset_name_from_type(self.datasetManager.type, subname="") in i]
        self.total_len = 0
        for file_name_2 in files: self.total_len += len(np.load(self.datasetManager.datasets_directory + '/' + file_name_2, allow_pickle=True))

        count = 0
        for file_name in files:

            if(type==1): self.simulate_on_dataset(file_name=file_name, visualise=visualise)
            if(type==2): self.simulate_on_dataset_2(file_name=file_name, visualise=visualise)
            if visualise: print(
                "\nFinished dataset '{name}' | {count}/{len}".format(name=file_name, count=count + 1, len=len(files)))

            count += 1

    def save_actions_stats(self):
        if(self.trained_testing): train_state = "Trained"
        else: train_state = "Training"
        name = "{dir}/{train_state}/{model_type}/{subname1}".format(dir=self.actions_stats_dir, train_state=train_state, model_type=self.model_type, subname1=self.model_subname)
        if not os.path.exists(name): os.mkdir(name)
        name+="/{subname2}.npy".format(subname2=self.model_subname2)
        np.save(name, np.array(self.model_same_action))

    def get_state_row_data(self, row, dim=[], vid_inputs=[]):

        if dim==[]: dim = self.dim
        if vid_inputs==[]: vid_inputs = self.vid_inputs

        num_data = []
        vid_data = []
        state = []
        if (dim[0] != 0): num_data.append(row[4][0]) #np.array #tf.convert_to_tensor(row[4][0], dtype=tf.int64) #np.asarray(row[4][0]).astype(np.float32)
        if (dim[1] != 0): num_data.append(row[5])
        if (dim[2] != 0): num_data.append(row[7])

        if (vid_inputs[0] != 0): vid_data.append(row[1][:,:,0])
        if (vid_inputs[1] != 0): vid_data.append(row[1][:,:,1])
        if (vid_inputs[2] != 0): vid_data.append(row[1][:,:,2])
        if (vid_inputs[3] != 0): vid_data.append(row[2])
        if (vid_inputs[4] != 0): vid_data.append(row[3])
        if (vid_inputs[5] != 0): vid_data.append(self.datasetManager.preprocessManager.resized(row[9])[0])

        if vid_data!=[]:vid_data=np.array([vid_data])
        for num_element in num_data: state.append(np.array([num_element]))
        state.append(vid_data)

        if len(state)==0: state=state[0]

        # Convert action to model action using bidict
        action = self.actions_relations.inverse[row[5]]
        reward = row[6]

        return state, action, reward

    def convert_propabilitites(self, action_probs):
        #print(action_probs)
        if np.amin(action_probs)<0:
            for i in range(len(action_probs)):
                action_probs[i] = action_probs[i]+abs(np.amin(action_probs))
        propabilities = []
        sum = np.sum(action_probs)
        for propability in action_probs:
            #print("propability: {propability}| sum: {sum}".format(propability=propability, sum=sum))
            propabilities.append(propability / sum)

        return np.array(propabilities) #.astype(int)

    def test(self):
        num_data=[]
        vid_data=[]
        x=[]
        if(self.dim[0] != 0): num_data.append(self.datasetManager.dataset[:, 4])
        if(self.dim[1] != 0): num_data.append(self.datasetManager.dataset[:, 5])
        if(self.dim[2] != 0): num_data.append(self.datasetManager.dataset[:, 7])
        if(len(num_data) != 0): x.append(num_data)

        if(self.vid_inputs[0] != 0): vid_data.append(self.datasetManager.dataset[:, 1])
        if(self.vid_inputs[1] != 0): vid_data.append(self.datasetManager.dataset[:, 2])
        if(self.vid_inputs[2] != 0): vid_data.append(self.datasetManager.dataset[:, 3])
        if(len(vid_data) != 0): x.append(vid_data)

        y = np.array([len(self.datasetManager.dataset), self.output_size]).fill(0)
        #y[]
        #self.Agent.Model.model.fit(x=x, y=[np.asarray(self.datasetManager.dataset[:,5]).astype(int)], epochs=200, batch_size=8) #np.int
