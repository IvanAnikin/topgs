import numpy as np

masks = {
    'green': (
        np.array([68, 124, 170]), np.array([99, 255, 255]) # [59, 106, 171], [94, 255, 255]
    ),
    'red': (
        # np.array([25, 0, 50]), np.array([169, 255, 255])    # use only one range of red
        np.array([0, 169, 200]), np.array([25, 255, 255])
    ),
    'blue': (
        # np.array([0, 0, 231]), np.array([102, 255, 255])
        np.array([93, 22, 226]), np.array([125, 255, 255])
    )
}