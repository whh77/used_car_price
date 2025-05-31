import os
import pandas as pd
import numpy as np
import subprocess
from feature.preprocess import preprocess_data
from model.train import train_model, predict

def main():
    # 加载数据
    train_path = '../data/used_car_train_20200313.csv'
    test_path = '../data/used_car_testA_20200313.csv'

    train = pd.read_csv(train_path, sep=' ')
    test = pd.read_csv(test_path, sep=' ')

    # 数据预处理
    train = preprocess_data(train)
    test = preprocess_data(test)

    # 统计均值编码映射
    mean_brand = train.groupby('brand')['price'].mean()
    mean_model = train.groupby('model')['price'].mean()
    test['brand_mean_price'] = test['brand'].map(mean_brand)
    test['model_mean_price'] = test['model'].map(mean_model)

    # 删除仅训练集有的列前，先保存 price
    y_train = train['price']
    drop_cols = ['price', 'price_bin']
    train = train.drop(columns=[col for col in drop_cols if col in train.columns])
    test = test.drop(columns=[col for col in drop_cols if col in test.columns])

    # 保证特征顺序一致
    feature_cols = sorted(train.columns)
    train = train[feature_cols]
    test = test[feature_cols]

    # 模型训练和预测
    model = train_model(train, y_train)
    predictions = predict(model, test)

    # 保存预测结果（按照样例格式要求）
    os.makedirs('../prediction_result', exist_ok=True)
    predictions.to_csv('../prediction_result/predictions.csv',
                       index=False,
                       columns=['SaleID', 'price'])
    print("预测结果已按照样例格式保存至 prediction_result/predictions.csv")



if __name__ == "__main__":
    main()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_submit_path = os.path.join(script_dir, '../prediction_result/generate_submit.py')
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    subprocess.run(['python', gen_submit_path], check=True, cwd=project_root)