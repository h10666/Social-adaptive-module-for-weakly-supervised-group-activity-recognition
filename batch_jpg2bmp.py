from PIL import Image
import time
import os

def batch_jpg2bmp(source_root, save_root):
    count = 0
    for root, dirs, files in os.walk(source_root):
        for img in files:
            if img.endswith('.jpg'):
                save_path = os.path.join(save_root, '/'.join((root.split('/')[-2:])))
                if not(os.path.exists(save_path)):
                    os.makedirs(save_path)
                jpg_img = Image.open(os.path.join(root,img)).resize((224,224)).save(
                    os.path.join(save_path, os.path.splitext(img)[0]+".bmp"))
                count+=1
                if count%720==0:
                    print(count//72)
    print('total:', count//72)
    
if __name__ == '__main__':
    batch_jpg2bmp("/mnt/ssd/yanrui/dataset/BD_v2_3/videos/", "/mnt/ssd/yanrui/dataset/BD_v2_3/videos_bmp/")