import json
import os


with open('../ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json', 'r') as f: #label文件
    label_raw_val = json.load(f)
    
    
label_raw = []
 
def file_name2(file_dir):   #特定类型的文件
    L=[]   
    image = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':   
                L.append(os.path.join(root, file))
                image.append(file)
                label_raw.append({'image_id':file, 'label_id':1})
    return L, image

path, image_id = file_name2('/home/wayne/python/kaggle/Ai_challenger/classification/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922') #图片目录


with open('scene_test_annotations.json', 'w') as f:
    json.dump(label_raw, f)