# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 01:53:00 2024

@author: bruce
"""
import os
import sys
import pandas as pd
import shutil

#Use This code to get command line arguments
#args = sys.argv[1:]
#Define the image directory and renamed image directory
#png_image_dir=args[0]
#png_image_dir_rn=args[1]

#Use This code for testing
png_image_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_dir' 
png_image_dir_rn='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_dir_rn'
png_image_aux_dir='c:\\Users\\bruce\\Desktop\\GAVRT\\GAVRT_20240722\\png_image_aux_dir'

#make a list of image files
files = os.listdir(png_image_dir)

#Copy files in png_image_dir to png_image_dir_rn
for file_name in files:
    shutil.copy(png_image_dir+'\\'+file_name, png_image_dir_rn+'\\'+file_name)

#create a blank scanid set. No dupes in sets
scanids=set()

#Loop through png file names and add scanids into set
for file_name in os.listdir(png_image_dir_rn):
    scanids.add(int(file_name[0:4]))

#Sort Scanids and convert to a set
scanids_list=list(sorted(scanids))

#Compute Seqno vector
seqnos_list=range(1,len(scanids)+1,1)

#Combine scanids and seqno vector into dictionary
scanid_dictionary=dict(zip(scanids_list,seqnos_list))

#save the dictionary to Excel
df=pd.DataFrame(scanid_dictionary.items(),columns=["scanid", "seqno"])

#Output scanid to dictionary to Excel
df.to_excel(png_image_aux_dir+'\\scanid_dictionary.xlsx','data',index=False)

#Loop Through file_names and rename 
for file_name in os.listdir(png_image_dir_rn):
    
    #Compute file_name and file_channel
    file_channel=file_name[9:11]
    file_scanid=int(file_name[0:4])
    
    #Compute image placement which is based on cahnnel
    if file_channel=='02':
        image_suffix='-01'
        
    elif file_channel=='04':
        image_suffix='-02'
        
    elif file_channel=='06':
        image_suffix='-03'
        
    elif file_channel=='08':
        image_suffix='-04'
        
    elif file_channel=='10':
        image_suffix='-05'
        
    elif file_channel=='12':
        image_suffix='-06'
        
    elif file_channel=='14':
        image_suffix='-07'  
    
    elif file_channel=='16':
        image_suffix='-08'
        
    else:
        image_suffix=='-XX'

    print('Old Filename: ',png_image_dir_rn+'\\'+file_name)
    print('New Filename: ',png_image_dir_rn+'\\'+'SD'+str(scanid_dictionary[file_scanid])+image_suffix+'.png')
    print('\n')
    
    os.rename(png_image_dir_rn+'\\'+file_name,png_image_dir_rn+'\\'+'SD'+str(scanid_dictionary[file_scanid])+image_suffix+'.png')

#Move NoImage.png to the Rename folder
#os.system('copy '+args[0]+'\\NoImage.png'+''+args[1]+'\\NoImage.png')

print(scanids)
