from os import listdir
import pandas as pd
import os.path as osp
import mmcv


# make fake test_label.csv
test_list = sorted(listdir("./dataset/test_image")) 


class_list = [0]* len(test_list)
df = pd.DataFrame({'File': test_list,
                   'Class': class_list,
                   'X1': class_list,
                  'Y1': class_list,
                  'Y1': class_list,
                  'X2': class_list,
                  'Y2': class_list,
                  'X3': class_list,
                  'Y3': class_list,
                  'X4': class_list,
                  'Y4': class_list,})
df.to_csv('./dataset/test_label.csv', index=False)  




import pandas as pd

#make fake test_label.json from test_label.csv
BINARY = True

# test
def convert_143_to_coco(ann_file, out_file, image_prefix):
    
    with open(ann_file, 'r') as l:
        data_infos = pd.read_csv(l)

    
    image_name = (data_infos['File'].unique())
    image_num = (len(image_name))
        
    annotations = []
    images = []
    obj_count = 0
    
    for idx, filename in enumerate((image_name)):
        if idx %50 ==0:
            print(f"{idx}/{image_num}")
        
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        data_per_filename = data_infos[data_infos.File == filename]
        for _,obj in data_per_filename.iterrows():  #index, obj
            px = [obj['X1'],obj['X2'],obj['X3'],obj['X4']]
            py = [obj['Y1'],obj['Y2'],obj['Y3'],obj['Y4']]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)] #0.5를 왜 더해주지?
            poly = [p for x in poly for p in x]
            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))

            # image_id는 이미지
            # id는 말 그대로 object에 할당된 id
            # category_id >> label
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0 if BINARY else 1,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0) # category_id >> obj['Class']

            annotations.append(data_anno)
            obj_count += 1

         
           

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': '0'}] if BINARY else [{'id':1, 'name': '1'},{'id':2, 'name': '2'},{'id':3, 'name': '3'},{'id':4, 'name': '4'},{'id':5, 'name': '5'},{'id':6, 'name': '6'},{'id':7, 'name': '7'}]) #  [{'id':1, 'name': '1'}, ......]
    if BINARY:
        mmcv.dump(coco_format_json, out_file+'.json')
    else:
        mmcv.dump(coco_format_json, out_file+'_multi'+'.json')

    
convert_143_to_coco('./dataset/test_label.csv', './dataset/test_label', './dataset/test_image')