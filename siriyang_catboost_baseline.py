#!/usr/bin/env python
# coding: utf-8

# <font color='red'>使用前请创建相应的文件夹或修改文件路径，为防止提交相同的结果被官方封号，至少请修改随机数种子以后再训练提交！！！</font>
# 
# 目录结构：
# ```
# .
# ├── dataset # 原始数据集
# │   ├── entprise_evaluate.csv
# │   ├── entprise_submit.csv
# │   ├── train
# │   │   ├── annual_report_info.csv
# │   │   ├── base_info.csv
# │   │   ├── change_info.csv
# │   │   ├── entprise_info.csv
# │   │   ├── news_info.csv
# │   │   ├── other_info.csv
# │   │   └── tax_info.csv
# │   └── train.zip
# ├── feature # 特征工程中间文件
# │   ├── cat_cols.json 
# │   ├── test.csv 
# │   └── train.csv 
# ├── importance.csv 
# ├── result # 提交结果
# │   └── cat_sub.csv 
# └── siriyang_catboost_baseline.ipynb
# ```

# # 特征提取

# ## 读取原始数据集和导包

# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json

base_info = pd.read_csv("./train/base_info.csv")
annual_report_info = pd.read_csv("./train/annual_report_info.csv")
tax_info = pd.read_csv("./train/tax_info.csv")
change_info = pd.read_csv("./train/change_info.csv")
news_info = pd.read_csv("./train/news_info.csv")
other_info = pd.read_csv("./train/other_info.csv")
train_label = pd.read_csv("./train/entprise_info.csv")
test = pd.read_csv("./entprise_evaluate.csv")


# ## 特征工程

# In[4]:


def get_feature():
    
    cat_cols=[
    'base_oplocdistrict',
    'base_industryco',
    'base_enttype',
    'base_enttypeitem',
    'base_state',
    'base_orgid',
    'base_jobid',
    'base_adbusign',
    'base_townsign',
    'base_regtype',
    'base_compform',
    'base_venind',
    'base_enttypeminu',
    'base_protype',
    'base_enttypegb',
    ]
    
    # -------------------------提取base_info表特征---------------------------
    
    col=['oplocdistrict','industryco','enttype','enttypeitem',
        'state','orgid','jobid','adbusign','townsign','regtype','empnum',
        'compform','parnum','exenum','venind','enttypeminu','protype',
        'regcap','reccap','forreccap','forregcap','congro','enttypegb']
    feature=base_info[['id']+col]
    feature.rename(columns={i:'base_'+i for i in col },inplace=True)
    
    # 将industryphy字段进行编码
    id_to_base_industryphy=list(set(base_info['industryphy']))
    base_industryphy_to_id={k:v+1 for v,k in enumerate(id_to_base_industryphy)}
    feature['base_industryphy_id']=base_info['industryphy'].map(lambda x:base_industryphy_to_id[x])
    cat_cols.append('base_industryphy_id')
    
    # -------------------------提取annual_report_info表特征---------------------------
    
    
    # -------------------------提取tax_info表特征---------------------------

    
    # -------------------------提取news_info表特征---------------------------
    
    
    # -------------------------提取other_info表特征---------------------------
    
    
    print(feature)
    
    for i in cat_cols:
        print(i)
        
    # 将分类特征的字段名保存，以便后期训练时读取
    with open('./feature/cat_cols.json', 'w') as fw:
        # 将字典转化为字符串
        # json_str = json.dumps(data)
        # fw.write(json_str)
        # 上面两句等同于下面这句
        json.dump(cat_cols,fw)
    
    return feature


# In[6]:



if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    
    
    feature=get_feature()
    
    train_feature = train_label.merge(feature, how='left', on='id', copy=False)
    test_feature = test.merge(feature, how='left', on='id', copy=False)
    
    print(train_feature)
    print(test_feature)
    for i in feature.columns:
        print(i)
    
    train_feature.to_csv('./feature/train.csv',index=False)
    test_feature.to_csv('./feature/test.csv',index=False)
        
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d s' % (int((datetime.now() - start).seconds)))


# # 模型训练

# ## 读取构建好的特征、分类字段名和导包

# In[9]:


import numpy as np
import pandas as pd
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

train = pd.read_csv('./feature/train.csv')
test = pd.read_csv('./feature/test.csv')
sub = pd.read_csv("./entprise_submit.csv")

train.fillna(0,inplace=True)
test.fillna(0,inplace=True)

feat = [c for c in train.columns if c not in ['id','label']]

cat_cols=[]
with open('./feature/cat_cols.json','r') as f:
        cat_cols = json.load(f)
for i in cat_cols:
        print(i)

for i in train.columns:
    if i in cat_cols:
        train[i] = train[i].astype('str')
        test[i] = test[i].astype('str')


# ## 训练

# In[10]:



if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    
    num_folds=5
    kfold = StratifiedKFold(n_splits=num_folds, random_state=1024, shuffle=False).split(train.drop(['label'],axis=1), train['label'])

    oof_probs = np.zeros(train.shape[0])
    output_probs = np.zeros((test.shape[0],5))
    offline_score = []
    feature_importance_df = pd.DataFrame()
    for fold, (train_idx, valid_idx) in enumerate(kfold):
        X_train, y_train = train[feat].iloc[train_idx], train['label'].iloc[train_idx]
        X_valid, y_valid = train[feat].iloc[valid_idx], train['label'].iloc[valid_idx]
        
        model=CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            task_type="CPU",
            learning_rate=0.01,
            iterations=10000,
            random_seed=2020,
            od_type="Iter",
            depth=8,
            early_stopping_rounds=500,

        )

        clf = model.fit(X_train,y_train, eval_set=(X_valid,y_valid),verbose=500,cat_features=cat_cols)
        yy_pred_valid=clf.predict(X_valid)
        y_pred_valid = clf.predict(X_valid,prediction_type='Probability')[:,-1]
        oof_probs[valid_idx] = y_pred_valid
        offline_score.append(f1_score(y_valid, yy_pred_valid))
        output_probs[:, fold] = clf.predict(test[feat],prediction_type='Probability')[:,-1]
        
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = model.feature_names_
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('OOF-MEAN-F1:%.6f, OOF-STD-F1:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    feature_importance_df_ = feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False)
    # print(feature_importance_df_.head(15))
    print(feature_importance_df_)
    feature_importance_df_.to_csv("./importance.csv")
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d s' % (int((datetime.now() - start).seconds)))


# ## 保存结果

# In[12]:


sub['score'] = np.mean(output_probs, axis=1)
print(sub['score'])
sub.to_csv("./result/cat_sub.csv")


# In[ ]:




