import os
import scipy.ndimage
from shutil import copyfile

input_dataset_path = 'raw_data'
output_dataset_path = 'clean_data'
if not os.path.exists(output_dataset_path):
    os.makedirs(output_dataset_path)

categories = ["baseball", "cricket"]
for c in categories:
    if not os.path.exists(os.path.join(output_dataset_path,c)):
        os.makedirs(os.path.join(output_dataset_path,c))

categories = ["baseball", "cricket"]

for c in categories:
    to_remove = []
    for im_id in os.listdir(os.path.join(input_dataset_path,c)):
        im_path = os.path.join(input_dataset_path,c,im_id)
        try:
            im = scipy.ndimage.imread(im_path)
            scipy.misc.imsave(im_path, im)
        except:
            to_remove.append(im_id)
            print("Error reading or writing image {}".format(im_path))
    
    for im_id in os.listdir(os.path.join(input_dataset_path,c)):
        if im_id not in to_remove:
            copyfile(os.path.join(input_dataset_path,c, im_id), os.path.join(output_dataset_path,c, im_id))