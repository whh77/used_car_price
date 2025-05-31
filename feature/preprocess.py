import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def preprocess_data(df):
    df = df.copy()

    # 1. 修复损坏字段
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    df['notRepairedDamage'] = df['notRepairedDamage'].astype(float)
    df['notRepairedDamage'] = df['notRepairedDamage'].fillna(df['notRepairedDamage'].median())

    # 2. 日期处理
    df['regDate'] = pd.to_datetime(df['regDate'], format='%Y%m%d', errors='coerce')
    df['creatDate'] = pd.to_datetime(df['creatDate'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['regDate', 'creatDate'])
    df['used_time'] = (df['creatDate'] - df['regDate']).dt.days
    df['reg_year'] = df['regDate'].dt.year
    df['reg_month'] = df['regDate'].dt.month
    df['reg_day'] = df['regDate'].dt.day
    df['creat_year'] = df['creatDate'].dt.year
    df['creat_month'] = df['creatDate'].dt.month
    df['creat_day'] = df['creatDate'].dt.day

    # 3. 缺失值处理
    for col in ['bodyType', 'fuelType', 'gearbox']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. 异常值处理
    df = df[(df['used_time'] >= 0) & (df['used_time'] < 365*30)]  # 排除异常使用年限
    if 'power' in df.columns:
        df.loc[df['power'] > 600, 'power'] = 600  # 限制最大马力

    # 5. 类别编码
    for col in ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 6. 新特征：价格分组（仅训练集有price时可用）
    if 'price' in df.columns:
        df['price_bin'] = pd.qcut(df['price'], 5, labels=False, duplicates='drop')

    # 7. 新特征：注册到创建的月份数
    df['used_months'] = ((df['creatDate'].dt.year - df['regDate'].dt.year) * 12 +
                         (df['creatDate'].dt.month - df['regDate'].dt.month))

    # 8. 新特征：品牌/车型的均值编码（仅训练集有price时可用）
    if 'price' in df.columns:
        for col in ['brand', 'model']:
            mean_map = df.groupby(col)['price'].mean()
            df[f'{col}_mean_price'] = df[col].map(mean_map)

    return df