import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Загрузка заранее обработанных птиц
fft_train = np.load("train_fft_new.npy")
names_train = np.load("name_of_birds.npy")
fft_test = np.load("test_fft_new.npy")
names_test = np.load("test_names.npy")

# Сплит данных, test_size - какая часть данных будет в y_train и y_test
# Сплит рандомно пихает элементы, можно задать сид через random_state
# X_train, X_test, y_train, y_test = train_test_split(fft, names, test_size=0.3)

# Расскомментить одну из строк, чтобы использовать
# Создать классификатор заново и обучить его
RFC_model = RandomForestClassifier(max_depth=5, n_estimators=193, criterion='entropy', max_features=3,
                                   max_leaf_nodes=12)
RFC_model.fit(fft_train, names_train.ravel())

# Использовать заранее обученный классификатор
# RFC_model = joblib.load("neuron.pkl")

RFC_predictions = RFC_model.predict(fft_test)

# Результаты тестов
# Сначала идет матрица, по диагонали правильные прогнозы, если в строке не на основной диагонали не нули,
# значит была ошибка прогнозирования, далее приводится более подробная статистика, затем итоговая точность
# модели на данном наборе данных
# Пример: 78  0  0
#          4 73  4
#          0  1 80
# Значит была ошибка в прогнозах соловья (4 раза сказал утка, 4 - синица), в синице один раз сказал, что соловей
# Насколько я заметил лучше всего утки работают, потом синицы, с соловьями чаще ошибки
# Погрешность предсказаний не более 8%, причем на большем объеме данных ошибка уменьшается
result = confusion_matrix(names_test, RFC_predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(names_test, RFC_predictions)
print("Classification Report:",)
print(result1)
result2 = accuracy_score(names_test, RFC_predictions)
print("Accuracy:", result2)
