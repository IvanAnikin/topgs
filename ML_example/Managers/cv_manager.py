
import cv2
import numpy as np
import os

class CV_Manager():

    def __init__(self, name = 'helmet', threshold=0.8, images_directory='../src/img/'):
        self.name = name
        self.good = []
        self.keypoints_1 = []
        self.keypoints_2 = []
        self.images_directory = images_directory
        self.ending = '.jpg'
        self.image_template = cv2.imread(self.images_directory + self.name + self.ending)
        self.threshold = threshold

        files = [i for i in os.listdir(self.images_directory) if
                 os.path.isfile(os.path.join(self.images_directory, i)) and name in i]

        self.image_templates = []
        for file_name in files:
            self.image_templates.append(cv2.imread(self.images_directory + file_name))

    def find_object_2(self, frame):
        object_found = False
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = self.image_template
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            object_found=True
        return frame, object_found


    def find_object(self, new_image, MIN_MATCH_COUNT=15, count=0):

        for image_template in self.image_templates:
            if(count == 0):
                # Function that compares input image to template
                # It then returns the number of SIFT matches between them
                image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)

                # Create SIFT detector object
                #sift = cv2.SIFT()
                sift = cv2.xfeatures2d.SIFT_create()
                # Obtain the keypoints and descriptors using SIFT
                self.keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
                self.keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

                # Define parameters for our Flann Matcher
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
                search_params = dict(checks = 100)

                # Create the Flann Matcher object
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                # Obtain matches using K-Nearest Neighbor Method
                # the result 'matchs' is the number of similar matches found in both images
                matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

                self.good = []
                # store all the good matches as per Lowe's ratio test.
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        self.good.append(m)

            object_found = False
            new_frame = new_image.copy()

            if len(self.good) > MIN_MATCH_COUNT:
                src_pts = np.float32([self.keypoints_1[m.queryIdx].pt for m in self.good]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.keypoints_2[m.trainIdx].pt for m in self.good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                new_frame = cv2.drawMatches(new_frame, self.keypoints_1, image_template, self.keypoints_2, self.good, None, **draw_params)

                object_found = True

                return new_frame, object_found

        return new_frame, object_found