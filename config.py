class Config:
    dataset_class = (
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    )
    n_class = 21


    # predict parameters
    pred_score_thresh = 0.9
    pred_nms_thresh = 0.3

    #
    weight_decay = 0.0005
    lr = 1e-3


cfg = Config()
