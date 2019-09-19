# 2019 SeedCup baseline

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
  ├── test.py
  └── train.py
  ```

### 网络结构
- embedding层 对id类数据做嵌入(embedding)
- FC层 对拼接后的向量做全连接处理

### 指标说明
- 误差率 RankScore
  - rankScore = MSE(real_signed_time ,pred_signed_time ) 精确到小时
  - MSE为均方根误差
- 准时率 onTimePercent
  - onTimePercent = Count(pred_singed_date <= real_signed_date)  精确到天
- 准确率 Accuracy
  - 精确到天,该指标仅供参考

### 问题反馈
- 请提issue或发邮件至`andrewzhou924@qq.com`