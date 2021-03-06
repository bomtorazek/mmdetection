_base_ = '../dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('1','2','3','4','5','6','7',) 
#classes =  ('0',)
data = dict(
    train=dict(
        img_prefix='./dataset/train_image',
        classes=classes,
        ann_file='./dataset/train_label_multi'),
    val=dict(
        img_prefix='./dataset/train_image',
        classes=classes,
        ann_file='./dataset/val_label_multi'),
    test=dict(
        img_prefix='./dataset/train_image',
        classes=classes,
        ann_file='./dataset/val_label_multi_1.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth'