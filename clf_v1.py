#数据源是IM+后台会话明细查询表里‘所有会话小结描述’ + ‘小结备注’
#目前仅可判断到三级分类，通过‘小结备注’ -> ‘所有会话小结描述’(目标变量)
#整体样本量不大，采用随机森林模型，目前精度不高，存在一些类别准确度为0，需要提升精度

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加载数据
data = pd.read_excel('D:\DA\python\流程AI\IM_chat.xlsx')

# 提取三级分类
def extract_third_level(desc):
    if not isinstance(desc, str):
        return '其他'
    parts = desc.split('>')
    return '>'.join(parts[:3]) if len(parts)>=3 else '其他'

data['三级分类'] = data['所有会话小结描述'].apply(extract_third_level)

# 合并低频类别
min_samples = 5
category_counts = data['三级分类'].value_counts()
valid_categories = category_counts[category_counts >= min_samples].index
data['三级分类_优化'] = data['三级分类'].where(
    data['三级分类'].isin(valid_categories), 
    '其他'
)

# 编码目标变量
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['三级分类_优化'])

# 文本清洗和向量化
def clean_text(text):
    text = str(text)
    text = re.sub(r'$$.*?$$', '', text)  # 移除所有[...]内容
    text = re.sub(r'\W+', ' ', text)     # 移除非字母数字字符
    return text.lower().strip()

data['清洗后备注'] = data['小结备注'].apply(clean_text)
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['清洗后备注'])

# 分层分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# 训练模型
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
present_labels = np.unique(y_test)
target_names = label_encoder.inverse_transform(present_labels)

print(classification_report(
    y_test, 
    y_pred, 
    target_names=target_names,
    labels=present_labels
))
