# model/train.py
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train):
    # 特征选择
    drop_cols = ['SaleID', 'regDate', 'creatDate']
    features = [col for col in X_train.columns if col not in drop_cols]

    X = X_train[features]
    y = np.log1p(y_train)  # log1p transform

    # 数据分割
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )

    # 类别特征支持
    categorical_features = [col for col in features if str(X_train[col].dtype) == 'int32' or str(X_train[col].dtype) == 'category']

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'max_depth': 8,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(100)
        ]
    )

    return model

def predict(model, test_data):
    # 特征选择(与训练时一致)
    drop_cols = ['SaleID', 'regDate', 'creatDate']
    features = [col for col in test_data.columns if col not in drop_cols]
    X_test = test_data[features]

    # 预测
    y_test_pred = model.predict(X_test)

    # 生成提交文件（按照样例格式）
    submission = pd.DataFrame({
        'SaleID': test_data['SaleID'].astype(int),
        'price': np.expm1(y_test_pred).round().astype(int)
    })

    return submission