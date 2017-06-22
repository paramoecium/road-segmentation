class Config():
    num_epochs = 3000
    validation_summary_frequency = 1000
    checkpoint_frequency = 1000
    batch_size = 32
    log_directory = 'logs/'
    examples_to_show = 5
    image_size = 400
    post_process_patch_size = 8
    train_image_size = image_size // post_process_patch_size
    train_size = 100
    dropout_train = 1.0
    learning_rate = 0.005
    ae_step = 4
    corruption = 0.01
