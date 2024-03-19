import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt

def generate_samples(NumSamples = 100):
    # аргументы точек на первой четверти окружности радиуса 2
    arguments = numpy.random.uniform(low=0, high=numpy.pi/2, size=(NumSamples, 1))
    # непосредственно точки
    v = 2 * numpy.cos (arguments)   # x - независимая переменная
    res = 2 * numpy.sin (arguments) # y - зависимая переменная

    return (v, res)


def draw_samples(v, res, style, title):
    plt.scatter(v, res, marker=style, label=title)
    plt.axis('square')
    plt.show()


def calculate_regression_by_standard_method(v, res):
    '''
    Выбран метод лассо-регрессии
    Документация: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

    Модель Lasso минимизирует значение:
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    где первое слагаемое - среднеквадратичное отклонение
    ||w||_1 - норма вектора коэффициентов модели в манхэттенской метрике,
    alpha - коэфф-т, ограничивающий рост коэффициентов модели

    не пойму - зачем они нарисовали 2 в знаменателе?
    '''
    model = linear_model.Lasso(alpha=0.1)
    model.fit(v, res)
    prediction = model.predict(v)

    return prediction

def draw_samples_and_prediction(v, res, prediction):

    plt.scatter(v, res, label='samples', marker='o')
    plt.scatter(v, prediction, label='prediction', marker='*')
    plt.axis('square')
    plt.show()

def calc_error(res, prediction):
    #среднеквадратичное отклонение
    return numpy.linalg.norm(res - prediction) / res.shape[0]

n_samples = 10
v, res = generate_samples(n_samples)
draw_samples(v, res, 'o', 'samples')
prediction = calculate_regression_by_standard_method(v, res)
draw_samples(v, prediction, '*', 'prediction')
draw_samples_and_prediction(v, res, prediction)


error = calc_error(res, prediction)
print('MSE is {:.3f}'.format(error))
