class Config(object):
    #dataset
    mean_pixel_values = [104.00699, 116.66877, 122.67892, 137.86]
    img_width = 384
    img_height = 384
    train_root = './data/train_images'
    valid_output_dir = 'valid_temp'
    resume = 'model.pth'

    # hyper parameters
    batch_size = 2
    num_workers = 0
    num_epochs = 500
    model_output = 'checkpoints'
    in_chs = 8
    num_classes = 3
