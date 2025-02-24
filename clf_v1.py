from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 加载数据集
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树模型
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 保存模型到文件
joblib.dump(clf, "iris_model.pkl")
print("模型已保存为 iris_model.pkl")

clf = joblib.load("iris_model.pkl")

# 从用户输入获取数据
sepal_length = float(input("请输入花萼长度: "))
sepal_width = float(input("请输入花萼宽度: "))
petal_length = float(input("请输入花瓣长度: "))
petal_width = float(input("请输入花瓣宽度: "))

# 转换为 NumPy 数组
new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

class_names = ["setosa", "versicolor", "virginica"]

# 进行预测
prediction = clf.predict(new_data)
print(f"预测类别是: {class_names[prediction[0]]}")