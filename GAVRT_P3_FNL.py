#Create Collage

import os
from PIL import Image
import pandas as pd

#Define the location of files
png_image_dir_rn='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_dir_rn' 
png_image_aux_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_aux_dir'
collage_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\collage_dir'

#Change to the directory png_image_aux_dir ad import the Excel File 
os.chdir(png_image_aux_dir)
df=pd.read_excel('scanid_dictionary.xlsx')
n=df['seqno'].max() #Number of Animation frames

#Define Function to create Collage
def create_collage(width, height, listofimages,collage_no):
    cols = 4
    rows = 2
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('RGB', (width, height))
    ims = []
    for p in listofimages:
        im = Image.open(p)
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    new_im.save(collage_dir+'\\Collage'+str(collage_no)+'.jpg')
    
#Change to png_image_dir_rn
os.chdir(png_image_dir_rn)

#Loop Through all days and times
for i in range(1,n+1,1):
    image_list=[]
    for j in range(1,9,1):
        new_entry='SD'+str(i)+'-0'+str(j)+'.png'
    
        # CHeck for file existance to add to collage
        if os.path.exists(new_entry) and j!=2:
            image_list.append(new_entry)
        else:
            image_list.append(png_image_aux_dir+'\\NoImage.png')
            
            
    print(image_list)
    #create_collage(2560,960, image_list,i)
    create_collage(1728,576, image_list,i)

