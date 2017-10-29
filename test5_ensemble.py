'''
1,3: 95.84 [0.95668]
'''

import json
import pickle
import copy


phases = ['test', 'val'] #['test', 'val']
files = {'val':['submit/resnet152_places_softmax1_val.txt',
                'submit/resnet152_places_softmax3_val.txt']
    ,   'test':['submit/resnet152_places_softmax1_test.txt',
                'submit/resnet152_places_softmax3_test.txt']
    }




my_aug_softmax = {}
temp = {}

'''小心字典的深拷贝'''
for phase in phases:
    myid = len(files[phase])
    for it in range(myid):
        with open(files[phase][it], 'rb') as handle:
              temp[phase] = pickle.loads(handle.read())
              my_aug_softmax[str(it)] = copy.deepcopy(temp)

with open('data/test/scene_test_annotations.json', 'r') as f: #label文件, 测试的是我自己生成的
    label_raw_test = json.load(f)
with open('data/validation/scene_validation_annotations_20170908.json', 'r') as f: #label文件
    label_raw_val = json.load(f)

'''
ensemble
'''

def np_to_list(array):
    if(len(array)!=80):
        print(array)
    return ((-array).argsort()[:3]).astype('int32').tolist()
    
for phase in phases:
    final_results = []
    
    final_softmax = copy.deepcopy(my_aug_softmax['0'][phase]) 
    for key in final_softmax:
        final_softmax[key] *= 0

    if phase=='test':
        image_ids = label_raw_test #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": 5}]
    else:
        image_ids = label_raw_val
    
    for item in image_ids:
        image_id = item['image_id']
        for it in range(myid):
            final_softmax[image_id] += my_aug_softmax[str(it)][phase][image_id]
        final_softmax[image_id] /= len(range(myid))    
        
    with open(('submit/softmax5_%s.txt'% phase), 'wb') as handle:
        pickle.dump(final_softmax, handle)
    
    results = [] #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": [5,18,32]}]
    dict_ = {}
    for item in image_ids:
        dict_ ['image_id'] = item['image_id']
        
        dict_['label_id'] = np_to_list(final_softmax[item['image_id']])
        results.append(dict_)
        dict_ = {}
 
    with open(('submit/submit5_%s.json'% phase), 'w') as f:
        json.dump(results, f)
