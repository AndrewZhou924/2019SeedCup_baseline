import torch
import time
import os

'''
config for train and test period
'''
class Config(object):
    def __init__(self):
        self.USE_CUDA           =       torch.cuda.is_available()
        self.NUM_EPOCHS         =       1000

        self.TRAIN_BATCH_SIZE   =       64
        self.VAL_BATCH_SIZE     =       64
        self.TEST_BATCH_SIZE    =       64
        self.TRAIN_FILE         =       './data/SeedCup_pre_train.csv'
        self.VAL_FILE           =       './data/SeedCup_pre_train.csv'
        self.TEST_FILE          =       './data/SeedCup_pre_test.csv'
        self.TEST_OUTPUT_FOLDER =       './test_output/'
        self.TEST_OUTPUT_PATH   =       self.TEST_OUTPUT_FOLDER + 'test_' + str(int(time.time())) + '.txt'

        self.MODEL_FILE_NAME    =       'model.pkl'
        self.MODEL_SAVE_FOLDER  =       './model/'
        self.MODEL_SAVE_PATH    =       self.MODEL_SAVE_FOLDER + self.MODEL_FILE_NAME
        self.LR                 =       1e-3  # default learning rate

        self.EMBEDDING_DIM      =       100
        self.LINER_HID_SIZE     =       1024
        self.INPUT_SIZE         =       11
        
        self.OUTPUT_DIM              =   1
        self.OUTPUT_TIME_INTERVAL_2  =   1
        self.OUTPUT_TIME_INTERVAL_3  =   1
        self.OUTPUT_TIME_INTERVAL_4  =   1
        self.LOSS_1_WEIGHT           =   1

        self.uid_range            =   1505257
        self.plat_form_range      =   4
        self.biz_type_range       =   6
        self.product_id_range     =   51805   
        self.cate1_id_range       =   25
        self.cate2_id_range       =   244
        self.cate3_id_range       =   1429
        self.seller_uid_range     =   1000 
        self.company_name_range   =   929
        self.rvcr_prov_name_range =   31
        self.rvcr_city_name_range =   370

        self.val_step             =   1
        self.Dataset_Normorlize   =   False
        self.Train_Val_ratio      =   0.9
        self.Train_rankScore_threshold = 70
        self.Train_onTimePercent_threshold = 0.90

        self.mkdir()

    '''
    you can set your own learning rate strategy here
    '''
    def get_lr(self, epoch):

        # if (epoch+1) % 10 == 0 and self.LR > 1e-4:
        #     self.LR*=0.1
        # self.LR*=0.1

        print("epoch {}: learning rate {}".format(epoch, self.LR))
        return self.LR

    '''
    make diretory
    '''
    def mkdir(self):
        if not os.path.exists(self.TEST_OUTPUT_FOLDER):
            os.mkdir(self.TEST_OUTPUT_FOLDER)
        if not os.path.exists(self.MODEL_SAVE_FOLDER):
            os.mkdir(self.MODEL_SAVE_FOLDER)