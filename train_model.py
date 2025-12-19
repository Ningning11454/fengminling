# 模型训练代码（保存为 train_model.py 运行）
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# 加载数据（指定编码，解决UnicodeDecodeError）
try:
    # 优先尝试常用中文编码
    df = pd.read_csv('insurance-chinese.csv', encoding='gbk')
except:
    try:
        df = pd.read_csv('insurance-chinese.csv', encoding='gb2312')
    except:
        # 若仍报错，尝试utf-8（适用于已转码的文件）
        df = pd.read_csv('insurance-chinese.csv', encoding='utf-8')

# 数据预处理（与Web应用编码逻辑一致）
# 性别编码
df['sex_female'] = (df['性别'] == '女性').astype(int)
df['sex_male'] = (df['性别'] == '男性').astype(int)

# 吸烟状态编码
df['smoke_yes'] = (df['是否吸烟'] == '是').astype(int)
df['smoke_no'] = (df['是否吸烟'] == '否').astype(int)

# 区域编码
df['region_northeast'] = (df['区域'] == '东北部').astype(int)
df['region_southeast'] = (df['区域'] == '东南部').astype(int)
df['region_northwest'] = (df['区域'] == '西北部').astype(int)
df['region_southwest'] = (df['区域'] == '西南部').astype(int)

# 选择特征和目标变量
features = ['年龄', 'BMI', '子女数量', 'sex_female', 'sex_male', 
            'smoke_no', 'smoke_yes', 'region_northeast', 'region_southeast',
            'region_northwest', 'region_southwest']
X = df[features]
y = df['医疗费用']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林回归模型
rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
rfr_model.fit(X_train, y_train)

# 保存模型
with open('rfr_model.pkl', 'wb') as f:
    pickle.dump(rfr_model, f)

print("模型训练完成并保存为 rfr_model.pkl")
