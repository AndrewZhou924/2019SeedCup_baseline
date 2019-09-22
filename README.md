# [2019 SeedCup baseline](<https://github.com/AndrewZhou924/2019SeedCup_baseline>)

### 工程结构
- `config.py` : 工程参数配置
- `dataLoader.py` : 加载训练集及验证集
- `model.py` : 模型网络结构
- `evaluation` : 评测接口,可计算rankScore,MSE及accuracy
- `train.py` : 模型训练
- `test.py`  : 模型在测试集上进行预测,输出预测结果
- `data/` 存放数据集csv文件
- `model/` 存放模型文件
- `test_output` 存放测试集预测输出文件

- 工程树状图
  ```
  ├── config.py
  ├── data
  │   ├── SeedCup_pre_test.csv
  │   └── SeedCup_pre_train.csv
  ├── dataLoader.py
  ├── evaluation.py
  ├── model.py
  ├── README.md
  ├── requirement.txt
  ├── test.py
  └── train.py
  
  ```
### 运行环境

- python 3.6.8
- pytorch 1.1.0
- numpy 1.17.2
- tqdm 4.31.1
- CUDA 10.1
- cudnn 7.5

### 使用方法

- 配置运行环境，可使用pip或其他包管理工具进行安装
  - 参考安装方式：`pip3 install -r requirement.txt`
- 在工程根目录下新建data文件夹，将初赛数据集csv文件放入
- 训练，模型保存在`model/` 中
  - 执行`python3 train.py`
- 测试，测试集预测结果保存在`test_output/` 中
  - 执行`python3 test.py`

### 网络结构

- embedding层 对id类数据做嵌入(embedding)
- FC层 对拼接后的向量做全连接处理

### 评测指标说明
- 误差率 RankScore
  - rankScore = MSE(real_signed_time ,pred_signed_time ) 精确到小时
  - MSE为均方根误差
- 准时率 onTimePercent
  - onTimePercent = Count(pred_singed_date <= real_signed_date)  精确到天
- 准确率 Accuracy
  - 精确到天,该指标仅供参考

### 参赛选手可以考虑的改善之处

- 增加Tensorboard可视化训练过程
- 增加加载模型继续训练代码
- 重新设计loss函数
- 调整学习率策略
- 做多任务，预测多个时间

### 问题反馈
- 请提issue或发邮件至`andrewzhou924@qq.com`