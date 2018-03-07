from math import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import f as ff
import random

# Импорт данных
f = open('num4.txt', 'r')
inp = np.array([i.split() for i in f])
x = np.array([])
y = np.array([])
z = np.array([])

def sred(lst):
   return float(sum(lst)) / len(lst)

j = 0
for i in inp:
    x = np.append(x, float(i[0]))
    y = np.append(y, float(i[1]))
    # z = np.append(z, float(i[2]))
    j += 0.01 + random.uniform(-0.1, 0.1)
    z = np.append(z, j)

# Объем выборки
n = len(x)

# Полином
matXY = np.array([[1, i, j, i*j, pow(i, 2), pow(j, 2), pow(i, 3), pow(j, 3)] for i, j in zip(x, y)])
matZ = z.transpose()
buf = np.linalg.inv(np.dot(matXY.transpose(), matXY))
w = np.dot(np.dot(buf, matXY.transpose()), matZ)

# Линейка
matXYl = np.array([[1, i, j] for i, j in zip(x, y)])
bufl = np.linalg.inv(np.dot(matXYl.transpose(), matXYl))
wl = np.dot(np.dot(bufl, matXYl.transpose()), matZ)

# Функция анализа
def analysis(wan):
    # Вектор прогнозов
    if len(wan) <= 3:
        zanl = np.array([wan[0] + wan[1]*i + wan[2]*j for i, j in zip(x, y)])
    else:
        zanl = np.array([wan[0] + wan[1] * i + wan[2] * j + w[3]*i*j
         + w[4]*np.power(i, 2) + w[5]*np.power(j, 2)
         + w[6]*np.power(i, 3) + w[7]*np.power(j, 3) for i, j in zip(x, y)])
    # Ошибки
    e = np.array([i-j for i, j in zip(z, zanl)])
    # Оценка дисперсии ошибок
    s2 = np.dot(e.transpose(), e) / (n - (2+1))
    # Отклонение от среднего
    otkl = np.array([i - sred(z) for i in z])
    # Коэф детерминации
    r2 = 1 - np.dot(e.transpose(), e) / np.dot(otkl.transpose(), otkl)
    # Частные коэф эластичности
    ex = wl[1] * (sred(x)/sred(zanl))
    ey = wl[2] * (sred(y)/sred(zanl))

    # Ср кв отклонение
    def srkv (lst):
        sum = 0
        sr = sred(lst)
        for i in lst:
            sum += pow(i - sr, 2)
        return pow(sum/n, 1/2)

    # Частные бета коэф
    bx = wl[1] * (srkv(x)/srkv(z))
    by = wl[2] * (srkv(y)/srkv(z))

    # F-тест
    k = len(wl)-1
    ftest = (r2/(1 - r2)) * ((n - k - 1)/k)
    fkr = ff.isf(0.05, k, n-k-1)

    #RMSE
    for i, j in zip(zanl, z):
        rmse = 0
        rmse += pow(i - j, 2)
        pow(rmse/n, 1/2)

    #Мультиколлинеарность
    xy = np.array([])
    for i, j in zip (x, y):
        xy = np.append(xy, i*j)

    multic = (sred(xy) - (sred(x) * sred(y))) / (srkv(x)*srkv(y))

    print('Оценка дисперсии ошибок: ' + str(s2))
    print('Коэф детерминации: ' + str(r2))
    print('Коэф эластичности z/x: ' + str(ex))
    print('Коэф эластичности z/y: ' + str(ey))
    print('Бета коэф z/x: ' + str(bx))
    print('Бета коэф z/y: ' + str(by))
    print('F теста: ' + str(ftest))
    print('F критич: ' + str(fkr))
    print('RMSE: ' + str(rmse))
    print('Мультикол: ' + str(multic))

    if ftest > fkr:
        print('Уравнение адекватно!')
    else:
        print('Уравнение нельзя считать адекватным!')

# Анализ линейки и полинома
print('АНАЛИЗ ЛИНЕЙКИ')
analysis(wl)
print()
print('АНАЛИЗ ПОЛИНОМА')
analysis(w)

# Plot outputs

# Проекции
# cset = axes.scatter(x, y, 0, zdir='z', c='grey')
# cset = axes.scatter(x, z, 0, zdir='y', c='grey')
# cset = axes.scatter(y, z, 0, zdir='x', c='grey')

# Размножение x и y
i = np.arange(min(x)-0.08, max(x)+0.08, 0.1)
j = np.arange(min(y)-0.08, max(y)+0.08, 0.1)
xregr, yregr = np.meshgrid(i, j)

# Регрессионная функция
# Полином
zregr = [w[0] + w[1]*i + w[2]*j + w[3]*i*j
         + w[4]*np.power(i, 2) + w[5]*np.power(j, 2)
         + w[6]*np.power(i, 3) + w[7]*np.power(j, 3) for i, j in zip(xregr, yregr)]
#Линейка
zregrl = [wl[0] + wl[1]*i + wl[2]*j for i, j in zip(xregr, yregr)]

fig = plt.figure(1)
axes = Axes3D(fig)
axes.scatter(x, y, z)

fig2 = plt.figure(2)
axes2 = Axes3D(fig2)
axes2.scatter(x, y, z)

axes.set_xlabel('X')
axes.set_xlim(xmin=0)
axes.set_ylabel('Y')
axes.set_ylim(ymin=0)
axes.set_zlabel('Z')
axes.set_zlim(zmin=0)

axes2.set_xlabel('X')
axes2.set_xlim(xmin=0)
axes2.set_ylabel('Y')
axes2.set_ylim(ymin=0)
axes2.set_zlabel('Z')
axes2.set_zlim(zmin=0)

axes.plot_surface(xregr, yregr, zregrl, rstride=15, cstride=15, alpha=0.1)
axes2.plot_surface(xregr, yregr, zregr, rstride=15, cstride=15, alpha=0.1)
plt.show()
f.close()
