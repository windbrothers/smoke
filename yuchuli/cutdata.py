# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:03:37 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:11:32 2020

@author: 张严风
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import os

def photo_name_Get(path):
    i=0
    fileList = os.listdir(path)
    # print(fileList)
    for photo_Name in fileList: 
        i=i+1
#        print('正在执行')
#        reszie(photo_Name)
        
        cutpath='./cutdata2/'+photo_Name+'/'
        print(cutpath)
        isExists=os.path.exists(cutpath)
        print(isExists)
        if not isExists:
            os.mkdir(cutpath) 
        cut_image(path,photo_Name,cutpath)
    return photo_Name
def cut_image(path,photo_Name,cutpath):
    img_path=path+photo_Name
    image = cv2.imread(img_path)
    copy_img=image.copy()
    print(img_path)
    print(image.shape)
    k=100
    Id=0
    for i in range(10):
        for j in range(10):
            box=(k*i,k*j,k*(i+1),k*(j+1))
#            print(box)
            img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#             region = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            region = img.crop(box)
            print(region)

#            GG=cutpath+str(Id)+'_'+photo_Name
            GG='cutdata2/cutpath/'+str(Id)+'_'+photo_Name
 
            region = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2BGR)
            cv2.imwrite(GG,region)
            h=k*i+50
            w=k*j+50   
            num=str(Id)
            cv2.putText(copy_img, num, (h, w), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(0, 0, 255), 2)
            Id=Id+1
    draw_name_path='./tmp/'+photo_Name
    cv2.imwrite(draw_name_path,copy_img)
#    print(Id-1)
def reszie(photo_Name):
    img_path=path+photo_Name
    print(img_path)
    image = cv2.imread(img_path)
#    print(image)
    image1=str(image)
    print(image1)
    if(image1==None):
        print('error')
    else:
        photo = cv2.resize(image, (1000, 1000))
    ##    smokephoto
        resize_name_path='./nosmokephoto/nosmoke_'+photo_Name
        cv2.imwrite(resize_name_path,photo)
if __name__ == '__main__':
    path = './nosmokephoto/'
    photo_Name=photo_name_Get(path)
#    reszie(photo_Name)
    print("完成")
    
   

