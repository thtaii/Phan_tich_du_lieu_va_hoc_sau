import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Định nghĩa hàm load_data() và hiển thị 10 dòng đầu tiên
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Thay đổi đường dẫn file train.csv của bạn tại đây
df = load_data(r"D:\Lab04\titanic_disaster.csv")
print("--- 10 dòng đầu tiên của dữ liệu ---")
print(df.head(10))

# 2. Thống kê dữ liệu thiếu bằng Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Biểu đồ nhiệt (Heatmap) thể hiện dữ liệu thiếu")
plt.show()

print("\n--- Nhận xét về tình trạng thiếu dữ liệu ---")
print(df[["Age", "Cabin", "Embarked"]].isnull().sum())
print(
    "> Nhận xét: Biến 'Age' thiếu một lượng vừa phải (~20%). Biến 'Cabin' thiếu rất nghiêm trọng (hơn 75%). Biến 'Embarked' chỉ thiếu rất ít (2 dòng)."
)

# 3. Xử lý cột Name thành firstName và secondName
# Cấu trúc cột Name thông thường: "Braund, Mr. Owen Harris" -> Họ, Danh xưng. Tên chính
# Tách thành firstName (Họ) và secondName (Tên sau dấu phẩy bao gồm danh xưng)
df[["firstName", "secondName"]] = df["Name"].str.split(",", expand=True, n=1)
df["secondName"] = df["secondName"].str.strip()
df.drop(columns=["Name"], inplace=True)

# 4. Xử lý rút gọn kích thước cột Sex
df["Sex"] = df["Sex"].map({"male": "M", "female": "F"})

# 5. Xử lý dữ liệu thiếu trên biến Age
# a. Vẽ Box plot theo nhóm Pclass để xem phân phối tuổi
plt.figure(figsize=(8, 5))
sns.boxplot(x="Pclass", y="Age", data=df)
plt.title("Biểu đồ Boxplot tuổi theo từng hạng vé (Pclass)")
plt.show()

print("\n--- Nhận xét về tuổi trung bình giữa các nhóm Pclass ---")
print(df.groupby("Pclass")["Age"].median())
print(
    "> Nhận xét: Hành khách ở hạng vé cao hơn (Pclass=1) có độ tuổi trung bình lớn hơn hành khách ở hạng vé thấp hơn."
)
print(
    "> Quyết định: Thay thế giá trị Age thiếu bằng giá trị trung vị (hoặc trung bình) của từng nhóm Pclass sẽ chính xác hơn dùng trung bình toàn bộ."
)

# b. Tiến hành thay thế giá trị Age thiếu theo từng nhóm Pclass
df["Age"] = df.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Kiểm tra lại bằng Heatmap sau khi xử lý Age
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Heatmap sau khi đã xử lý thiếu dữ liệu cột 'Age'")
plt.show()

# 6. Xây dựng biến Agegroup
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

# Tách danh xưng từ secondName (ví dụ: "Mr. Owen Harris" -> "Mr.")
df["namePrefix"] = df["secondName"].str.split(".").str[0]
df["secondName"] = df["secondName"].apply(
    lambda x: ".".join(x.split(".")[1:]).strip()
)

# 8. Khai thác thông tin familySize
df["familySize"] = 1 + df["SibSp"] + df["Parch"]

# 9. Tạo đặc trưng 'Alone'
df["Alone"] = df["familySize"].apply(lambda x: 1 if x == 1 else 0)

# 10. Tiến hành tách loại cabin (typeCabin)
df["Cabin"] = df["Cabin"].fillna("Unknown")
df["typeCabin"] = df["Cabin"].astype(str).str[0]
