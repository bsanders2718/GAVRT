#-CREATE GIF AND MP4 ANimations

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

#----FIX THIS to ADD GIF AND MP4 INPUT LOCATIONS------#

#Define the location of files
png_image_aux_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_aux_dir'
collage_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\collage_dir'
animations_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\animations_dir'

#Change to the directory png_image_aux_dir ad import the Excel File 
os.chdir(png_image_aux_dir)
df=pd.read_excel('scanid_dictionary.xlsx')
n=df['seqno'].max() #Number of Animation frames

#Change to the directory collage_dir 
os.chdir(collage_dir)

# Create GIF Animation
nframes = n # Number of frames
plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

def animate(i):
    im = plt.imread('Collage'+str(i+1)+'.jpg')
    plt.imshow(im)

anim = FuncAnimation(plt.gcf(), animate, frames=nframes,interval=1000)
anim.save('solar_animation_gif.gif', writer='Pillow')

#create MP4 Solar Animation File .MP4 Format
os.system("ffmpeg -framerate 1 -i Collage%d.jpg -c:v libx264 -r 30 solar_animation_mp4.mp4")

#Move animations to animation folder
shutil.move(collage_dir+'\\solar_animation_mp4.mp4',animations_dir+'\\solar_animation_mp4.mp4')
shutil.move(collage_dir+'\\solar_animation_gif.gif',animations_dir+'\\solar_animation_gif.gif')