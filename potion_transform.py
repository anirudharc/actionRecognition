import cv2
import os
import numpy as np

def load_images_from_folder(path):
    images = []

    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)
        # img = cv2.threshold(img, 1, 255, cv2.THRESH_TRUNC)

        if img is not None:
            potion_t = img.copy()
            potion_t[np.where(potion_t > [0])] = [255]
            potion_t = cv2.cvtColor(potion_t, cv2.COLOR_GRAY2BGR)
            potion_t[np.where((potion_t == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

            images.append(potion_t)

    agg_image = images[0]
    for idx, img in enumerate(images[1:]):
        agg_image = cv2.add(agg_image, img)
        name = "/media/bighdd1/arayasam/actionRecognition/output/agg_image_" + str(idx) + ".png" 
        cv2.imwrite(name, img)

    return images, agg_image

im_list, agg_image = load_images_from_folder("/media/bighdd1/arayasam/pose_model/output_heatmaps_folder/")
cv2.imwrite("/media/bighdd1/arayasam/actionRecognition/output/agg_image.png", agg_image)
