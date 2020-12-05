import pandas as pd
import os.path as osp
import mmcv

BINARY = True

def convert_143_to_coco(ann_file, out_file, image_prefix, fold):
    
    with open(ann_file, 'r') as l:
        data_infos = pd.read_csv(l)
        

    
    # for i in range(len(data_infos['File'])):
    #     if data_infos['File'][i] == '378.tif':
    #         data_infos['File'][i] = '328.tif'
    #         if data_infos['File'][i+1] != '378.tif':
    #             break
    
    image_name = (data_infos['File'].unique())
    image_num = (len(image_name))
    
    train_idx = [(fold-1)%5,(fold)%5,(fold+1)%5,(fold+2)%5] 
    val_idx = [(fold+3)%5]
    
    
    
    annotations_train = []
    images_train = []
    obj_count_train = 0
    
    annotations_val = []
    images_val = []
    obj_count_val = 0
    
    for idx, filename in enumerate((image_name)):
        if idx %50 ==0:
            print(f"{idx}/{image_num}")
        
        if idx%5 in train_idx:
            
            img_path = osp.join(image_prefix, filename)
            height, width = mmcv.imread(img_path).shape[:2]

            images_train.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))
        
            data_per_filename = data_infos[data_infos.File == filename]
            for _,obj in data_per_filename.iterrows():  #index, obj
                px = [obj['X1'],obj['X2'],obj['X3'],obj['X4']]
                py = [obj['Y1'],obj['Y2'],obj['Y3'],obj['Y4']]
                poly = [(x , y ) for x, y in zip(px, py)] #0.5를 왜 더해주지?
                poly = [p for x in poly for p in x]
                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))
                
                # image_id는 이미지
                # id는 말 그대로 object에 할당된 id
                # category_id >> label
                data_anno = dict(
                    image_id=idx,
                    id=obj_count_train,
                    category_id=0 if BINARY else obj['Class'],
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0) # category_id >> obj['Class']

                annotations_train.append(data_anno)
                obj_count_train += 1
        
        elif idx%5 in val_idx:
            img_path = osp.join(image_prefix, filename)
            height, width = mmcv.imread(img_path).shape[:2]

            images_val.append(dict(
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
                    id=obj_count_val,
                    category_id=0 if BINARY else obj['Class'],
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0) # category_id >> obj['Class']

                annotations_val.append(data_anno)
                obj_count_val += 1
            
           

    coco_format_json_train = dict(
        images=images_train,
        annotations=annotations_train,
        categories=[{'id':0, 'name': '0'}] if BINARY else [{'id':1, 'name': '1'},{'id':2, 'name': '2'},{'id':3, 'name': '3'},{'id':4, 'name': '4'},{'id':5, 'name': '5'},{'id':6, 'name': '6'},{'id':7, 'name': '7'},]) #  [{'id':1, 'name': '1'}, ......]
    if BINARY:
        mmcv.dump(coco_format_json_train, out_file+'_'+str(fold)+'.json')
    else:
        mmcv.dump(coco_format_json_train, out_file+'_multi'+'_'+str(fold)+'.json')
    coco_format_json_val = dict(
        images=images_val,
        annotations=annotations_val,
        categories=[{'id':0, 'name': '0'}] if BINARY else [{'id':1, 'name': '1'},{'id':2, 'name': '2'},{'id':3, 'name': '3'},{'id':4, 'name': '4'},{'id':5, 'name': '5'},{'id':6, 'name': '6'},{'id':7, 'name': '7'}]) #  [{'id':1, 'name': '1'}, ......]
    
    if BINARY:
        mmcv.dump(coco_format_json_val, out_file.replace('train','val')+'_'+str(fold)+'.json')
    else:
        mmcv.dump(coco_format_json_val, out_file.replace('train','val')+'_multi'+'_'+str(fold)+'.json')
convert_143_to_coco('./dataset/train_label.csv', './dataset/train_label', './dataset/train_image', fold = 5)