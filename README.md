# This Project Contains Animation Code written in Python for GAVRT Radio Astronomy Facility

Animation Process Directories and Python Code
Bruce M. Sanders 07/24/2024


Directories:
•	scan_input_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\scan_input_dir'  # Directory that has the raster scan data

•	scan_output_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\scan_output_dir' # Directory that the raster scan data is moved to after processed

•	png_image_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_dir' Directory that the resulting png images are deposited.

•	Png_image_dir_rn: 'c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_dir_rn' . Renamed png_images.

•	png_image_aux_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_aux_dir' #directory that has the YAML file and the NoImage.png file which is a blank image used for Channel 2

•	collage_dir: C:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\collage_dir

•	animations_dir:  C:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\animations_dir

 
Python Code:
GAVRT_P1_FNL.py: Loops through scan files in the directory scan_input_dir and creates photos in the directory png_image_dir. Processed scan files are moved to scan_output_dir- Done
 
GAVRT_P2_FNL.py: Copies images in png_images_dir to png_images_dir_rn. Then renames the images for creating collage and animation sequences. -Done
 
GAVRT_P3_FNL.py: Uses renamed images in png_images_dir_rn to create collage images. Stores Collage images in collage directory. Uses NoImage.png in png_images_aux_dir for Channel 2-Done
 
GAVRT_P4_FNL.py: Uses images in collage directory to create MP4 and GIF Animations

