from math import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker
from scipy.stats import t
import random

# Импорт данных
f = open('num4.txt', 'r')
inp = np.array([i.split() for i in f])
x = np.array([])
y = np.array([])

for i in inp:
    x = np.append(x, float(i[1]))
    y = np.append(y, float(i[0]))

# y = [pow(i, 2) + random.uniform(-0.15, 0.15) for i in x]

# Объем выборки
n = len(x)

a = np.array([[1, i, pow(i, 2), pow(i, 3)] for i in x])
b = np.dot(a.transpose(), a)
w = np.dot(np.linalg.inv(b), (np.dot(a.transpose(), y)))

a2 = np.array([[1, i] for i in x])
b2 = np.dot(a2.transpose(), a2)
w2 = np.dot(np.linalg.inv(b2), (np.dot(a2.transpose(), y)))

yreg1 = np.array([(w[0] + w[1] * i + w[2] * pow(i, 2) + w[3] * pow(i, 3)) for i in x])
yreg2 = np.array([(w2[0] + w2[1] * i) for i in x])

# Суммы квадратов регрессионных остатков
r = [pow(i - j, 2) for i, j in zip(y, yreg1)]
r2 = [pow(i - j, 2) for i, j in zip(y, yreg2)]
sse = sum(r)
sse2 = sum(r2)

# Стандартное отклонение
sb = pow((n * sse) / (n * sum([pow(i, 2) for i in x]) - pow(sum(x), 2)), 1/2)
sb2 = pow((n * sse2) / (n * sum([pow(i, 2) for i in x]) - pow(sum(x), 2)), 1/2)

# Значение статистики критерия t
tpr = abs(w[1] / sb)
tpr2 = abs(w2[1] / sb2)

# Стьюдент
tst = t.isf(0.2, n-4)
tst2 = t.isf(0.3, n-2)

# Доверительный интервал
ydov = [i - tst * sb for i in yreg1]
ydov2 = [i + tst * sb for i in yreg1]

ydov3 = [i - tst2 * sb2 for i in yreg2]
ydov32 = [i + tst2 * sb2 for i in yreg2]

# Вывод в консоль
print('ОЦЕНКА КАЧЕСТВА одномерной регрессии')
print('Суммы квадратов регресс остатков (SSE): ' + str(sse2))
print('Стандартное отклонение: ' + str(sb2))
print('Значение статистики критерия t: ' + str(tpr2))
print('Значение из табл Стьюдента (альфа=0.05): ' + str(tst2))
if tpr2 > tst2:
    print('Коэффициент регрессии является значимым!')
else:
    print('Коэффициент регрессии не является значимым!')

print()
print('ОЦЕНКА КАЧЕСТВА полиноминальной регрессии')
print('Суммы квадратов регресс остатков (SSE): ' + str(sse))
print('Стандартное отклонение: ' + str(sb))
print('Значение статистики критерия t: ' + str(tpr))
print('Значение из табл Стьюдента (альфа=0.05): ' + str(tst))
if tpr > tst:
    print('Коэффициент регрессии является значимым!')
else:
    print('Коэффициент регрессии не является значимым!')

# Plot outputs
figure = plt.figure()
axes = figure.add_subplot(1, 1, 1)
plt.figure(1)
plt.scatter(x, y, color='black', s=2)
plt.plot(x, ydov3, color='grey', linewidth=1)
plt.plot(x, ydov32, color='grey', linewidth=1)
plt.plot(x, yreg2, color='green', linewidth=1, label='$f(x) = a + bx$')
locator = matplotlib.ticker.MaxNLocator()
axes.xaxis.set_major_locator(locator)
axes.grid()
plt.legend()

figure = plt.figure(2)
axes2 = figure.add_subplot(1, 1, 1)
plt.figure(2)
plt.scatter(x, y, color='black', s=2)
plt.plot(x, ydov, color='grey', linewidth=1)
plt.plot(x, ydov2, color='grey', linewidth=1)
plt.plot(x, yreg1, color='blue', linewidth=1, label='$f(x) = a + bx + cx^2 + dx^3$')
locator2 = matplotlib.ticker.MaxNLocator()
axes2.xaxis.set_major_locator(locator2)
axes2.grid()
plt.legend()

plt.show()
f.close()
