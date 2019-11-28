import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

""" Từ dữ liệu ban đầu
    1. Đọc tất cả các câu hỏi từ file text loại bỏ dấu câu không cần thiết
    2. Đọc tất cả các nhãn tương ứng câu hỏi
        2.1 Lấy nhãn và gán bằng các giá trị (number) mặc định tự động
    3. Xây dựng dữ liệu train dùng KNN để phân lớp
    4. tính độ chính xác của model
    5. test
"""


def get_key(val):
    for key, value in dict.items():
        if val == value:
            return key


def label_code(input_):
    result = {}
    index = 0
    for i in range(len(input_)):
        if input_[i] not in result.keys():
            result.update({input_[i]: index})
            index += 1
    return result
    pass


def testing(input):
    vect = CountVectorizer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_ = vect.fit_transform(X)
    x_test = vect.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X_, y)
    y_pred = knn.predict(x_test)
    test = vect.transform([input])
    nearest_neighbor = df['QUES'][knn.kneighbors(test, 1)[1][0][0]]
    ans = df['ANS'][knn.kneighbors(test, 1)[1][0][0]]
    label = get_key(knn.predict(test)[0])
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return nearest_neighbor, ans, label, accuracy


if __name__ == '__main__':
    header = ['STT', 'QUES', 'ANS', 'LABEL']
    df = pd.read_csv('./all_data.csv', header=None, names=header)
    X = df['QUES']
    dict = label_code(df['LABEL'])
    y = df['LABEL_num'] = df.LABEL.map(dict)

    test = 'trường có ngành cntt không ?'

    nn, ans, label, acc = testing(test)
    print('Câu hỏi:', test)
    print('Nearest question: ', nn)
    print('Nhãn: ', label)
    print('Trả lời: ', ans)
    print('Độ chính xác: ', acc)
