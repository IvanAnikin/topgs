mport numpy as np from scipy.spatial import distance
class Tracker ():
def init_ (self, limit: list, max disappeared: int, max distance: int, max history int):
# Object Info
self.centroids = (0
self. colors self.bboxes • ()
self.disappeared • {
self.counted = ()
self.nextID = 0
# Tracking Info
self.max disappeared = max disappeared
self.max distance = max distance
self.max history = max history
# Counting Info
self. limit = limit
self.people in = 0
colf noonio out

def add object (self, centroid: tuple, bbox: list):
"'"Add an object to track.
Ares:
centroid (tuple): tuple with the (x,y) coordinates of the centroid bbox (list): list with four integer values: [left, top, right, bottom]
self.centroids[self.nextID] = [centroid]
self.colors [self.nextID] = tuple(
np.append (np.random. randint (255, size=(3)), 255))
self.disappeared[self.nextID] = 0
self.bboxes [self.nextI0] = bbox
self. counted[self.nextI0] = False
self.nextID
+= 1


def remove object (self, objectIO: int):
"''*Removes an object info based on its 10.
Ares:
objectID (int): ID of the object to remove
del self.centroids[objectID1 del self.colors [objectID]
del self.disappeared[objectID]
del self.counted obiectID1

def append centroid(self, centroid: tuple, bbox: list, objectID: int):
''''"Appends a new centroid to an existing object based on its ID.
Args:
centroid (tuple): tuple with the (x,y) coordinates of the centroid bbox (list): list with four integer values: [left, top, right, bottom] objectID (int): ID of the object
if len(self. centroids [objectID]) ›= self. max history
del self.centroids [objectID1[0]
self centroids [objectID]-append(centroid)
self.bboxes [objectID1 = bbox

def get min distance (self, centroid: tuple, available ids: list):
"''*Returns the ID of the object closest to the centroid, or -1 if no object is
close.
Args:
centroid (tuple): tuple with the (x,y) coordinates of the centroid available ids (list): list with the IDs of all available objects
MEET
min dist = self.max distance
min id = -1
for objectID in available ids:
last centroid = self. centroids [objectID][-1]
dist = distance. euclidean (list (centroid), list(last centroid))
if dist < min dist:
min dist = dist
min id = objectID
return min id

def update centroids (self, detections: list):
" Updates all centroids
Args:
detections (list): list of tuples (centroid, bbox) containing all detections.
centroid (tuple): tuple with the (x,y) coordinates of the centroid bbox (list): list with four integer values: [left, top, right, bottom]
available ids = list(self.centroids.keys ())
for (centroid, bbox) in detections:
id to update = self.get min distance(centroid, available ids)
if id to update != -1:
self.append centroid(centroid, bbox, id to update) available_ids.remove (id to _update)
else:
self.add object (centroid, bbox) if len(available ids) > 0:
for objectID in available ids:
self.disappeared[objectID] += 1
if self. disappeared [objectID] > self.max disappeared:
self.remove object (objectID


def update(self, detections: list):
"'''*Processes all detections, adding, updating, or deleting objects.
Angs:
detections (list): list of tuples (centroid, bbox) containing all detections.
centroid (tuple): tuple with the (x,y) coordinates of the centroid bbox (list): list with four integer values: [left, top, right, bottom]
if len(detections) == 0:
for objectID in list(self.disappeared.keys ()):
self.disappeared [objectID] += 1
if self.disappeared[objectID] > self .max disappeared:
self.remove object (objectID)
elif len(self…centroids) == 0:
for (centroid, bbox) in detections:
self. add object (centroid, bbox)
else:
self.update centroids (detections)

def count people (self):
"*'"'Counts people going in or out.
y = self.limit[0](1]
for centroid key in self.centroids. keys):
if self. counted[ centroid key] == False:
centroid hist = self. centroids [centroid key]
y mean = np.mean(centroid[11
for centroid in centroid hist [:-1]])
y last = centroid hist [-11[11
diff = y last - y mean
if diff > 0 and y last > y and y mean < y:
self. people out += 1
self.counted[centroid key] = True
elif diff < 0