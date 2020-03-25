from glob import glob
import cv2
import numpy
from tqdm import tqdm


label_files  = sorted(glob("/home/muromirg/datasets/final1_001/bar/masks_machine/*"))
label_files += sorted(glob("/home/muromirg/datasets/final1_001/lay/masks_machine/*"))
label_files += sorted(glob("/home/muromirg/datasets/final1_001/room_clothes/masks_machine/*"))

print(len(label_files))

for file in tqdm(label_files) :
    img = cv2.imread(file)
    img[img == 2] = 0
    cv2.imwrite(file, img)
