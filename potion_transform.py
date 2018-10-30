import cv2
import os
import numpy as np

def load_images_from_folder(path):
    images = []

    for filename in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            potion_t = img.copy()
            potion_t[np.where(potion_t > [0])] = [255]
            potion_t = cv2.cvtColor(potion_t, cv2.COLOR_GRAY2BGR)

            images.append(potion_t)

    potion = []
    agg_image = images[0]
    agg_image[np.where(True)] = [0, 0, 0]
    canvas_split = len(images)//2 + 1
    r = g = True
    b = False

    for idx, img in enumerate(images):

        ratio = (idx % canvas_split)/canvas_split
        if idx >= canvas_split:
            r = False
            b = True
            ratio = 1 - ratio
    
        print(idx, idx % canvas_split, ratio)
        img[np.where((img == [255, 255, 255]).all(axis=2))] = [b*(1 - ratio)*255, g*ratio*255, r*(1-ratio)*255]
        potion.append(img)

        agg_image = cv2.add(agg_image, img)
        name = "/media/bighdd1/arayasam/actionRecognition/output/agg_image_" + str(idx) + ".png" 
        cv2.imwrite(name, img)

    return images, agg_image

im_list, agg_image = load_images_from_folder("/media/bighdd1/arayasam/pose_model/output_heatmaps_folder/")
cv2.imwrite("/media/bighdd1/arayasam/actionRecognition/output/agg_image.png", agg_image)
