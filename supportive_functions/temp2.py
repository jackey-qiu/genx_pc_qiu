import os

image_path='M:\\fwog\\members\\qiu05\\raxr_scan_HfZr-S1-12h'
for folder in os.listdir(image_path):
    for file in os.listdir(os.path.join(image_path,folder)):
        original_file=os.path.join(os.path.join(image_path,folder),file)
        modified_file=os.path.join(os.path.join(image_path,folder),file.replace('raxr_scan_',''))
        os.rename(original_file,modified_file)
