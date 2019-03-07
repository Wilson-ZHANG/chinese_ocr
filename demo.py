#-*- coding:utf-8 -*-
import os
import ocr
import cv2
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
image_files = glob('./test_images/*.*')
#coors_files = glob('./test_coors/*.*')

def crop(image_path, coors, save_dir):
    image_name = os.path.split(image_path)[1].split('.')[0]
    image = cv2.imread(image_path)
    for index, coor in enumerate(coors):
        x1, y1, x2, y2 = coor[0], coor[1], coor[6], coor[7]
	#print(x1, y1, x2, y2)
        roi_image = image[y1: y2 + 1, x1: x2 + 1]
        cv2.imwrite(os.path.join(save_dir, image_name+'_{}.jpg'.format(index)), roi_image)


if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        t = time.time()
        result, image_framed , text_recs = ocr.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
#        crop(image_file,text_recs,'./test_coors')
       
