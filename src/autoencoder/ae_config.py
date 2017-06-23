class Config():
    num_epochs = 3000
    validation_summary_frequency = 1000
    checkpoint_frequency = 1000
    batch_size = 32
    log_directory = 'logs/'
    examples_to_show = 5
    train_image_size = 400 # train images are of size 400 x 400
    test_image_size = 608 # test images are of size 608 x 608
    cnn_pred_size = 8 # pixels in CNN prediction
    train_image_resize = train_image_size // cnn_pred_size # 50
    test_image_resize = test_image_size // cnn_pred_size # 76
    train_size = 100 # train data set size
    test_size = 50 # test data set size
    dropout_train = 1.0
    learning_rate = 0.005
    ae_step = 10
    corruption = 0.01
