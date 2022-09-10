import numpy as np
import cv2
def preprocess_test_image(img:np.array,min_size,):
    # Read in image from file path
    img=cv2.resize(img,(200,100),cv2.INTER_AREA)
    ret, bw = cv2.threshold(img, 190,255,cv2.THRESH_BINARY_INV)

    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    # min_size = 50 #threshhold value for small noisy components
    img2 = np.zeros((output.shape), np.uint8)

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    res = cv2.bitwise_not(img2)
    res=res/255.
    
    return res
