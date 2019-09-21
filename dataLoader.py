import csv
import torch
import numpy as np
import datetime
from config import Config
from torch.utils.data import Dataset
from tqdm import tqdm

'''
Dataset information
0 uid
1 plat_form
2 biz_type
3 create_time
4 payed_time

5 product_id
6 cate1_id
7 cate2_id
8 cate3_id
9 preselling_shipped_time
10 seller_uid

11 company_name
12 *lgst_company
13 *warehouse_id
14 *shipped_prov_id
15 *shipped_city_id

16 rvcr_prov_name
17 rvcr_city_name
18 *shipped_time
19 *got_time
20 *dlved_time
21 *signed_time
'''
class TrainSet(Dataset):
    def __init__(self, source_file, opt=None):

        with open(source_file, 'r') as f:
            i_range  = int((len(f.readlines()) - 1) * opt.Train_Val_ratio)
            self.len = i_range

        with open(source_file, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)[0]
            #  = np.array(list(reader)).shape[0] - 2
            self.inputs = []

            self.targets_sign_day  = []
            self.targets_sign_hour = []

            self.targets_ship_day  = []
            self.targets_ship_hour = []

            self.targets_got_day  = []
            self.targets_got_hour = []

            self.targets_dlved_day  = []
            self.targets_dlved_hour = []


            for i in tqdm(range(i_range)):
                data = next(reader)[0].split('\t')
                
                # build inputs and targets
                if opt.Dataset_Normorlize:
                    temp_input = [float(data[0])/opt.uid_range, float(data[1])/opt.plat_form_range, 
                                float(data[2])/opt.biz_type_range, float(data[5])/opt.product_id_range, 
                                float(data[6])/opt.cate1_id_range, float(data[7])/opt.cate2_id_range, 
                                float(data[8])/opt.cate3_id_range, float(data[10])/opt.seller_uid_range,
                                float(data[11])/opt.company_name_range, float(data[16])/opt.rvcr_prov_name_range, 
                                float(data[17])/opt.rvcr_city_name_range]
                else:
                    temp_input = [float(data[0]), float(data[1]), 
                                float(data[2]), float(data[5]), 
                                float(data[6]), float(data[7]), 
                                float(data[8]), float(data[10]),
                                float(data[11]), float(data[16]), 
                                float(data[17])]            
                temp_input = np.array(temp_input)

                # calculate time between payed_time and signed_time
                
                try:
                    temp_payed_time = datetime.datetime.strptime(data[4], "%Y-%m-%d %H:%M:%S")
                    temp_shipped_time = datetime.datetime.strptime(data[18], "%Y-%m-%d %H:%M:%S")
                    temp_got_time = datetime.datetime.strptime(data[19], "%Y-%m-%d %H:%M:%S")
                    temp_dlved_time = datetime.datetime.strptime(data[20], "%Y-%m-%d %H:%M:%S")
                    temp_signed_time = datetime.datetime.strptime(data[21], "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    continue

                time_interval_1 = temp_signed_time.day - temp_payed_time.day
                time_interval_2 = temp_signed_time.day - temp_shipped_time.day
                time_interval_3 = temp_signed_time.day - temp_got_time.day
                time_interval_4 = temp_signed_time.day - temp_dlved_time.day
            
                # remove noise
                if time_interval_1 <= 0 or time_interval_2 <= 0 or time_interval_3 <= 0 or time_interval_4 <= 0:
                    continue

                self.inputs.append(temp_input)

                self.targets_sign_day.append(time_interval_1)
                self.targets_sign_hour.append(temp_payed_time.hour)

                self.targets_ship_day.append(time_interval_2)
                self.targets_ship_hour.append(temp_shipped_time.hour)

                self.targets_got_day.append(time_interval_3)
                self.targets_got_hour.append(temp_got_time.hour)

                self.targets_dlved_day.append(time_interval_4)
                self.targets_dlved_hour.append(temp_dlved_time.hour)


    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets_sign_day = self.targets_sign_day[idx]
        targets_sign_hour = self.targets_sign_hour[idx]

        targets_ship_day = self.targets_ship_day[idx]
        targets_ship_hour = self.targets_ship_hour[idx]

        targets_got_day = self.targets_got_day[idx]
        targets_got_hour = self.targets_got_hour[idx]

        targets_dlved_day = self.targets_dlved_day[idx]
        targets_dlved_hour = self.targets_dlved_hour[idx]

        return (inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour)

    def __len__(self):
        return len(self.inputs)

class ValSet(Dataset):
    def __init__(self, source_file, opt=None):

        with open(source_file, 'r') as f:
            i_range  = len(f.readlines()) - 1
            self.len = i_range

        with open(source_file, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)[0]
            #  = np.array(list(reader)).shape[0] - 2
            self.inputs = []
            self.payed_time = []
            self.signed_time = []

            for i in tqdm(range(i_range)):
                data = next(reader)[0].split('\t')

                # avoid having same data with TrainSet
                if i < int(i_range * opt.Train_Val_ratio):
                    continue

                if opt.Dataset_Normorlize:
                    temp_input = [float(data[0])/opt.uid_range, float(data[1])/opt.plat_form_range, 
                                float(data[2])/opt.biz_type_range, float(data[5])/opt.product_id_range, 
                                float(data[6])/opt.cate1_id_range, float(data[7])/opt.cate2_id_range, 
                                float(data[8])/opt.cate3_id_range, float(data[10])/opt.seller_uid_range,
                                float(data[11])/opt.company_name_range, float(data[16])/opt.rvcr_prov_name_range, 
                                float(data[17])/opt.rvcr_city_name_range]
                else:
                    temp_input = [float(data[0]), float(data[1]), 
                                float(data[2]), float(data[5]), 
                                float(data[6]), float(data[7]), 
                                float(data[8]), float(data[10]),
                                float(data[11]), float(data[16]), 
                                float(data[17])]              

                temp_input = np.array(temp_input)

                self.inputs.append(temp_input)
                self.payed_time.append(data[4])
                self.signed_time.append(data[21])

            print("==> in ValSet, len(inputs)   is ", len(self.inputs))
            print("==> in ValSet, inputs.shape   is ", np.shape(self.inputs))


    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        payed_time = self.payed_time[idx]
        signed_time = self.signed_time[idx]
        return (inputs, payed_time, signed_time)

    def __len__(self):
        return len(self.inputs)

'''
available data fields in Testset
0 uid	
1 plat_form	
2 biz_type	
3 create_time	
4 payed_time	
5 product_id	
6 cate1_id	
7 cate2_id	
8 cate3_id	
9 preselling_shipped_time	
10 seller_uid	
11 company_name	
12 rvcr_prov_name	
13 rvcr_city_name
'''
class TestSet(Dataset):
    def __init__(self, source_file, opt=None):
        with open(source_file, 'r') as f:
            i_range  = len(f.readlines()) - 1
            self.len = i_range

        with open(source_file, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)[0]

            self.inputs = []
            self.payed_time = []

            for i in range(i_range):
                data = next(reader)[0].split('\t')

                if opt.Dataset_Normorlize:
                    temp_input = [float(data[0])/opt.uid_range, float(data[1])/opt.plat_form_range, 
                                float(data[2])/opt.biz_type_range, float(data[5])/opt.product_id_range, 
                                float(data[6])/opt.cate1_id_range, float(data[7])/opt.cate2_id_range, 
                                float(data[8])/opt.cate3_id_range, float(data[10])/opt.seller_uid_range,
                                float(data[11])/opt.company_name_range, float(data[12])/opt.rvcr_prov_name_range, 
                                float(data[13])/opt.rvcr_city_name_range]
                else:
                    temp_input = [float(data[0]), float(data[1]), 
                                float(data[2]), float(data[5]), 
                                float(data[6]), float(data[7]), 
                                float(data[8]), float(data[10]),
                                float(data[11]), float(data[12]), 
                                float(data[13])]              

                temp_input = np.array(temp_input)

                self.inputs.append(temp_input)
                self.payed_time.append(data[4])

            print("==> in TestSet, len(inputs)   is ", len(self.inputs))
            print("==> in TestSet, inputs.shape   is ", np.shape(self.inputs))


    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        payed_time = self.payed_time[idx]
        return (inputs, payed_time)

    def __len__(self):
        return len(self.inputs)
