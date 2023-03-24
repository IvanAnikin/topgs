import time
from termcolor import colored

import numpy as np
import cv2
import os

import PIL.Image as pil
from PIL import Image

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import matplotlib as mpl
import matplotlib.cm as cm

import gluoncv

class PreprocessManager():
    def __init__(self, contours_count = 5, minimal_distance = 15, type = "explorer", edges_avg = 120000, edges_max = 500000,
                 edges_coefficient = 20000, min_edges_sum_for_difference = 100, object_reward = 10, wanted_object_reward = 10,   new_object_reward = 10,
                 dim=[[0, 0, 1],[1, 1, 1, 1, 1, 1]], scale_percent=50, feed_width=96, feed_height=320, model_path="C:/Users/ivana/PycharmProjects/Machine_Learning/ComputerVision/AutopilotCar/Managers/CV_Models"):
        self.contours_count = contours_count
        self.minimal_distance = minimal_distance
        self.type = type
        self.edges_avg = edges_avg
        self.edges_max = edges_max
        self.edges_coefficient = edges_coefficient
        self.min_edges_sum_for_difference = min_edges_sum_for_difference
        self.model_path = model_path
        self.confidence_default = 0.5
        self.threshold_default = 0.3
        self.detected_objects = []
        self.new_object_reward = new_object_reward
        self.object_reward = object_reward
        self.wanted_object_reward = wanted_object_reward
        self.dim = dim
        self.scale_percent = scale_percent
        self.feed_width = feed_width
        self.feed_height = feed_height

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([self.model_path, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                   dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        self.weightsPath = os.path.sep.join([self.model_path, "yolov3.weights"])
        self.configPath = os.path.sep.join([self.model_path, "yolov3.cfg"])

    def canny_edges(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # grayscale frame
        return(cv2.Canny(frame_gray, 50, 100, 3))               # canny edges detection\

    def blackAndWhite(self, frame):
        resized, _ = self.resized(frame=frame)
        # gray and black and wite frames
        grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

        return blackAndWhiteImage

    def resized(self, frame):
        # reduce image size
        width = int(frame.shape[1] * self.scale_percent / 100)
        height = int(frame.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        return resized, dim

    def canny_and_contours(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(frame_gray, 50, 100, 3)

        ret, threshold = cv2.threshold(canny_edges, 170, 255, cv2.THRESH_BINARY)
        im, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # contours detection on thresholded cropped frame
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

        chosen_contours = []
        count = 0
        for c in cntsSorted:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, False)
            if area < 10001 and 100 < perimeter < 1000:
                if (count < self.contours_count):
                    chosen_contours.append(c)
                    count += 1

        return canny_edges, chosen_contours

    def reward_calculator(self, frame, last_frame, distance, canny_edges=[], objects=[], object_found=False):

        if canny_edges==[]: canny_edges=self.canny_edges(frame=frame)
        reward = 0

        # __1__ - Punishment for too little distance
        if (int(distance) <= self.minimal_distance): reward = reward - 10
        elif((self.type == "explorer" or self.type == "detective")):
            if(canny_edges.any() > 0):                   # reward for difference only if not too close distance
                edges_sum = 0
                for edges_row in canny_edges:
                    for edge in edges_row:
                        edges_sum += edge

                # __2__ - Frame difference reward
                # calculate frame difference from the previous frame if not too small edges sum                # -> difference from some ammount of the PREVIOUS/  (previous+next - unable for realtime training)
                if(edges_sum > self.min_edges_sum_for_difference):
                    difference = cv2.absdiff(frame, last_frame).astype(np.uint8)
                    difference = round(100 - (np.count_nonzero(difference) * 100) / difference.size, 2)  # 0 to 15
                    reward += difference

                # __3__ - Edges reward
                # Calculating edges count for higher reward for more edges
                edges_reward = edges_sum/self.edges_coefficient - self.edges_avg/self.edges_coefficient
                reward += edges_reward  # -> difference/different variable

            # __4__ - Objects reward
            # Reward for each object detected
            for object in objects:
                reward += self.object_reward
                if(object[0] not in self.detected_objects):
                    reward += self.new_object_reward
                    self.detected_objects.append(object[0])

            # __5__ - Reward for finding object it was looking for
            if(object_found): reward+=self.wanted_object_reward

        return reward

    def objects_detection(self, frame, visualise = False):

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        (W, H) = (None, None)

        objects_detected = []
        frame_start = time.time()
        frame, dim = self.resized(frame=frame)

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence_default:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_default,
                                self.threshold_default)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                objects_detected.append([classIDs[i], boxes[i]])

        if visualise:
            print("Frame timestep: {timestep} | net.forward timestep: {net_timestep} | Objects detected: {objects} | (W, H): ({W}, {H})".format(
                    timestep=np.round(time.time() - frame_start, 2), objects=objects_detected, W=W, H=H, net_timestep=end-start))
            cv2.imshow('frame after detection', frame)
            print()
            print(colored("Object found",'blue'))
            k = cv2.waitKey(1) & 0xff

        return objects_detected

    def monodepth(self, frame, visualise=False):

        # MOVE TO TRAIN AND SEND AS PARAM to make it faster --- ?
        # using cpu
        ctx = mx.cpu(0)
        model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_mono_640x192',
                                            # monodepth2_resnet18_kitti_stereo_640x192 monodepth2_resnet18_posenet_kitti_mono_640x192
                                            pretrained_base=False, ctx=ctx, pretrained=True)

        original_height, original_width = frame.shape[:2]

        raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(raw_img)

        img = img.resize((self.feed_width, self.feed_height), pil.LANCZOS)
        img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

        outputs = model.predict(img)
        disp = outputs[("disp", 0)]
        disp_resized = mx.nd.contrib.BilinearResize2D(disp, height=int(original_height),
                                                      width=int(original_width))

        disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()

        if (visualise):
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            depth_frame = np.asarray(im)
            output = np.concatenate((depth_frame, frame), axis=0)
            cv2.imshow('frame', output)

            k = cv2.waitKey(1)

            if k == 27:  # If escape was pressed exit
                cv2.destroyAllWindows()

        return disp_resized_np

    def state_preprocess(self, frame, distance=0):

        num_data = []
        vid_data = []
        state = []
        if (self.dim[0][2] != 0): num_data.append(distance)

        #if (self.dim[1][0] != 0 or self.dim[1][1] != 0 or self.dim[1][2] != 0):
        resized,dim=self.resized(frame=frame)
        if (self.dim[1][0] != 0): vid_data.append(resized[:, :, 0])
        if (self.dim[1][1] != 0): vid_data.append(resized[:, :, 1])
        if (self.dim[1][2] != 0): vid_data.append(resized[:, :, 2])
        if (self.dim[1][3] != 0): vid_data.append(cv2.resize(self.canny_edges(frame=frame), dim, interpolation=cv2.INTER_AREA))
        if (self.dim[1][4] != 0): vid_data.append(self.blackAndWhite(frame=frame))
        if (self.dim[1][5] != 0): vid_data.append(cv2.resize(self.monodepth(frame=frame), dim, interpolation=cv2.INTER_AREA))

        if vid_data != []: vid_data = np.array([vid_data])
        for num_element in num_data: state.append(np.array([num_element]))
        state.append(vid_data)

        if len(state) == 0: state = state[0]

        return state

