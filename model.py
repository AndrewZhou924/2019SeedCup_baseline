import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from config import Config
import numpy as np

'''
design loss function
'''
class My_MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        if x.sum() > y.sum():
            return torch.mean(torch.pow((x - y), 2))**0.5 * 10
        else:
            return torch.mean(torch.pow((x - y), 2))**0.5
            
class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        # self.encoder_uid       = nn.Embedding(opt.uid_range, opt.EMBEDDING_DIM)
        self.encoder_plat_form = nn.Embedding(opt.plat_form_range, opt.EMBEDDING_DIM)
        self.encoder_biz_type = nn.Embedding(opt.biz_type_range, opt.EMBEDDING_DIM)
        self.encoder_product_id = nn.Embedding(opt.product_id_range, opt.EMBEDDING_DIM)

        self.encoder_cate1_id = nn.Embedding(opt.cate1_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate2_id = nn.Embedding(opt.cate2_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate3_id = nn.Embedding(opt.cate3_id_range, opt.EMBEDDING_DIM)

        self.encoder_seller_uid = nn.Embedding(opt.seller_uid_range, opt.EMBEDDING_DIM)
        self.encoder_company_name = nn.Embedding(opt.company_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_prov_name = nn.Embedding(opt.rvcr_prov_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_city_name = nn.Embedding(opt.rvcr_city_name_range, opt.EMBEDDING_DIM)

        self.FC_1_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 1)
        )

        self.FC_1_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, 1)
        )

        self.FC_2_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_2)
        )

        self.FC_2_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_2)
        )


        self.FC_3_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_3)
        )

        self.FC_3_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_3)
        )

        self.FC_4_1 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_4)
        )

        self.FC_4_2 = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(10 * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_TIME_INTERVAL_4)
        )

    def forward(self, x):

        '''
        embedding layers
        '''
        output_encoder_plat_form = self.encoder_plat_form(x[:,1].long())
        output_encoder_biz_type = self.encoder_biz_type(x[:,2].long())
        output_encoder_product_id = self.encoder_product_id(x[:,3].long())

        output_encoder_cate1_id = self.encoder_cate1_id(x[:,4].long())
        output_encoder_cate2_id = self.encoder_cate2_id(x[:,5].long())
        output_encoder_cate3_id = self.encoder_cate3_id(x[:,6].long())
        output_encoder_seller_uid = self.encoder_seller_uid(x[:,7].long())

        output_encoder_company_name = self.encoder_company_name(x[:,8].long())
        output_encoder_rvcr_prov_name = self.encoder_rvcr_prov_name(x[:,9].long())
        output_encoder_rvcr_city_name = self.encoder_rvcr_city_name(x[:,10].long())

        concat_encoder_output = torch.cat((output_encoder_plat_form, 
        output_encoder_biz_type, output_encoder_product_id, 
        output_encoder_cate1_id, output_encoder_cate2_id,
        output_encoder_cate3_id, output_encoder_seller_uid,
        output_encoder_company_name, output_encoder_rvcr_prov_name,
        output_encoder_rvcr_city_name
        ), 1)

        '''
        Fully Connected layers
        you can attempt muti-task through uncommenting the following code and modifying related code in train()
        '''
        output_FC_1_1 = self.FC_1_1(concat_encoder_output)
        # output_FC_2_1 = self.FC_2_1(concat_encoder_output)
        # output_FC_3_1 = self.FC_3_1(concat_encoder_output)
        # output_FC_4_1 = self.FC_4_1(concat_encoder_output)

        output_FC_1_2 = self.FC_1_2(concat_encoder_output)
        # output_FC_2_2 = self.FC_2_1(concat_encoder_output)
        # output_FC_3_2 = self.FC_3_1(concat_encoder_output)
        # output_FC_4_2 = self.FC_4_1(concat_encoder_output)

        # return (output_FC_1_1, output_FC_2_1, output_FC_3_1, output_FC_4_1, output_FC_1_2, output_FC_2_2, output_FC_3_2, output_FC_4_2)
        return (output_FC_1_1, output_FC_1_2)