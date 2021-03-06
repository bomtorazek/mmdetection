_base_ = '../detectors/detectors_htc_r50_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('0',) # ('0',)
data = dict(
    train=dict(
        img_prefix='./dataset/train_image',
        classes=classes,
        ann_file='./dataset/train_label'),
    val=dict(
        img_prefix='./dataset/train_image',
        classes=classes,
        ann_file='./dataset/val_label'),
    test=dict(
        img_prefix='./dataset/test_image',
        classes=classes,
        ann_file='./dataset/test_label.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/detectors_htc_r50_1x_coco-329b1453.pth'