import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Загрузка заранее обработанных птиц
fft = np.load("fft_of_birds.npy")
names = np.load("name_of_birds.npy")

# Сплит данных, test_size - какая часть данных будет в y_train и y_test
# Сплит рандомно пихает элементы, можно задать сид через random_state
X_train, X_test, y_train, y_test = train_test_split(fft, names, test_size=0.2)

# Расскомментить одну из строк, чтобы использовать
# Создать классификатор заново и обучить его
# RFC_model = RandomForestClassifier()
# RFC_model.fit(X_train, y_train.ravel())

# Использовать заранее обученный классификатор
# RFC_model = joblib.load("neuron.pkl")

RFC_predictions = RFC_model.predict(X_test)

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
result = confusion_matrix(y_test, RFC_predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, RFC_predictions)
print("Classification Report:",)
print(result1)
result2 = accuracy_score(y_test, RFC_predictions)
print("Accuracy:", result2)
print(len(X_train), len(X_test))
