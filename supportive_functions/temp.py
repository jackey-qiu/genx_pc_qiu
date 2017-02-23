import os

image_path='M:\\fwog\\members\\qiu05\\raxr_scan_HfZr-S1-12h'
for folder in os.listdir(image_path):
    for file in os.listdir(os.path.join(image_path,file)):
        os.rename(file,file[10:])
