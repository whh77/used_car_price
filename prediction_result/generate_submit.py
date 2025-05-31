import os
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sample_path = 'data/used_car_sample_submit.csv'
pred_path = 'prediction_result/predictions.csv'
output_path = 'prediction_result/submit.csv'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 读取预测结果，构建 SaleID->price 映射
pred_df = pd.read_csv(pred_path)

pred_df['price'] = pred_df['price'].rolling(window=3, min_periods=1, center=True).mean().round().astype(int)
pred_dict = dict(zip(pred_df['SaleID'].astype(str), pred_df['price'].astype(str)))

# 按样例顺序输出
with open(sample_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        sale_id = row[0]
        price = pred_dict.get(sale_id, '0')
        writer.writerow([sale_id, price])