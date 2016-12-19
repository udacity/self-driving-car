class DataConfig(object):
    data_path = "/storage/hpc_tanel/sdc"
    data_name = "hsv_gray_diff_ch4"
    img_height = 192
    img_width = 256
    num_channels = 4
    
class TrainConfig(DataConfig):
    model_name = "comma_prelu"
    batch_size = 32
    num_epoch = 10
    val_part = 33
    X_train_mean_path = "data/X_mean_hsv_gray_diff_ch4_round2_val_part_33.npy"
    
class TestConfig(TrainConfig):
    model_path = "checkpoints/weights_hsv_gray_diff_ch4_comma_prelu_no_dropout-07-0.00318.hdf5"
    angle_train_mean = -0.004179079

class VisualizeConfig(object):
    pred_path = "submissions/submission_g_d2_n1_n2_co_e_phase2.csv"
    true_path = "data/CH2_final_evaluation.csv"
    img_path = "D:/udacity/phase2_test.tar/phase2_test/output_test/center/*.jpg"
