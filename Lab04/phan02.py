import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Tắt tất cả các cửa sổ biểu đồ cũ đang bị treo (nếu có)
plt.close("all")

df = pd.read_csv(r"D:\Lab04\titanic_disaster.csv")

# Xử lý các cột dữ liệu cần thiết cho các biểu đồ phía dưới
df["Sex"] = df["Sex"].map({"male": "M", "female": "F"})
df["familySize"] = 1 + df["SibSp"] + df["Parch"]

# Tạo đặc trưng Alone (Nếu familySize = 1 thì Alone = 1, ngược lại = 0)
df["Alone"] = df["familySize"].apply(lambda x: 1 if x == 1 else 0)

# Điền giá trị thiếu cho Age theo Pclass trung vị để phân loại nhóm tuổi
df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))


# Xây dựng biến Agegroup
def categorize_age(age):
    if age <= 12:
        return "Kid"
    elif age <= 18:
        return "Teen"
    elif age <= 60:
        return "Adult"
    else:
        return "Older"


df["Agegroup"] = df["Age"].apply(categorize_age)

# Điền giá trị thiếu cho cột Embarked
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# --- Câu 12: Tương quan tỉ lệ sống sót và thiệt mạng theo giới tính ---
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1")
plt.title("Tỉ lệ sống sót và thiệt mạng theo giới tính")
plt.xlabel("Giới tính (M: Nam, F: Nữ)")
plt.ylabel("Số lượng hành khách")
plt.legend(["Thiệt mạng", "Sống sót"])
plt.show()

# --- Câu 13: Tình hình hành khách sống sót trên từng Pclass ---
plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set2")
plt.title("Tình hình sống sót theo hạng vé (Pclass)")
plt.xlabel("Hạng vé")
plt.ylabel("Số lượng hành khách")
plt.legend(["Thiệt mạng", "Sống sót"])
plt.show()

# --- Câu 14: Tình hình sống sót trên từng nhóm giới tính và thang đo tuổi tác ---
sns.catplot(
    x="Agegroup",
    hue="Survived",
    col="Sex",
    data=df,
    kind="count",
    palette="pastel",
    order=["Kid", "Teen", "Adult", "Older"],
)
plt.subplots_adjust(top=0.8)
plt.suptitle("Sống sót theo nhóm tuổi và giới tính", y=0.98)
plt.show()

# --- Câu 15: Xác suất hành khách sống sót dựa trên Alone ---
plt.figure(figsize=(6, 4))
sns.barplot(x="Alone", y="Survived", data=df, errorbar=None, palette="coolwarm")
plt.title("Xác suất sống sót dựa trên việc đi một mình hay đi cùng gia đình")
plt.xticks([0, 1], ["Đi cùng gia đình", "Đi một mình"])
plt.ylabel("Xác suất sống sót")
plt.show()

# --- Câu 16: Xác suất hành khách sống sót dựa trên giá vé (Fare) ---
plt.figure(figsize=(10, 5))
sns.histplot(
    data=df, x="Fare", hue="Survived", multiple="stack", bins=40, kde=True
)
plt.xlim(0, 300)  # Giới hạn khung nhìn vé tránh các giá vé quá dị biệt làm loãng trục
plt.title("Phân phối và xác suất sống sót dựa trên giá vé (Fare)")
plt.xlabel("Giá vé")
plt.ylabel("Số lượng hành khách")
plt.show()

# --- Câu 17: Số lượng người thiệt mạng và sống sót theo Pclass và cảng cập bến ---
g = sns.FacetGrid(df, col="Embarked", row="Pclass", margin_titles=True)
g.map_dataframe(sns.countplot, x="Survived", palette="deep")
g.set_xticklabels(["Chết", "Sống"])
g.set_axis_labels("Trạng thái", "Số lượng")
plt.subplots_adjust(top=0.88)
g.fig.suptitle("Thống kê sống sót theo Hạng vé (Pclass) và Cảng cập bến")
plt.show()