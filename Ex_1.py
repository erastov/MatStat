from math import *
import pylab
import matplotlib.ticker
import scipy.stats
import numpy

f = open('num.txt', 'r')
a = [float(i) for i in f]

# Объем выборки
n = len(a)


# Функция просчета интервалов и частот
def progon(a, n):
    # Определяем шаг
    m = log2(n)
    h = (max(a) - min(a)) / m

    # Начальное значение интервалов
    start = min(a) - h / 2

    # Определяем интервалы и середины
    interv = []
    buf = start
    mid = [0] * (round(m) + 1)
    for i in range(0, round(m) + 1):
        interv.append([buf, buf + h])
        mid[i] = round(buf + (h / 2), 4)
        buf += h

    # Частота и частотность
    amou = [0] * (round(m) + 1)
    freque = [0] * (round(m) + 1)
    for i in a:
        key = True
        j = 0
        while key:
            if (i >= interv[j][0]) and (i < interv[j][1]):
                amou[j] += 1
                freque[j] = amou[j] / n
                key = False
            else:
                j += 1
    return interv, amou, freque, mid, h


interv = []
amou = []
freque = []
mid = []
h = 0

interv, amou, freque, mid, h = progon(a, n)

# Среднее
sred = 0
for fi, mi in zip(freque, mid):
    sred += fi * mi

# Выборочная дисперсия
disp = 0
for ai, mi in zip(amou, mid):
    disp += ai * pow((mi - sred), 2)
disp = disp / (n - 1)

# Ср кв отклонение
otkl = pow(disp, 1 / 2)

# Извлечение начал интервалов
b = [i[0] for i in interv]
d = [i[1] for i in interv]

# Эмпирическая функция распределения
c = [0 for i in freque]
j = 1
c[0] = freque[0]
for i in freque[1:]:
    c[j] = c[j - 1] + i
    j += 1

# Коэффициент вариации
if sred != 0:
    v = otkl / sred

# Выборочные начальные и центральные моменты
def moments(l):
    al = 0
    bl = 0
    for i, k in zip(amou, mid):
        al += i * pow(k, l)
        bl += i * pow(k - sred, l)
    al = al / n
    bl = bl / n
    return al, bl


a2, b2 = moments(2)
a3, b3 = moments(3)
a4, b4 = moments(4)

# Оценка коэффициента асимметрии
sk = b3 / pow(b2, 3 / 2)

# Оценка эксцесса
ex = b4 / pow(b2, 2) - 3

# Выборочная мода
ind = amou.index(max(amou))
vmoda = b[ind] + h * ((amou[ind] - amou[ind - 1]) / (2 * amou[ind] - amou[ind - 1] - amou[ind + 1]))


# Медианный интервал
def medinterv (amou):
    key = True
    i = 1
    summ = 0
    summ2 = 0
    while key:
        summ = sum([j for j in amou[:i]])
        if summ <= n / 2:
            summ2 = sum([j for j in amou[:i + 1]])
            if summ2 > n / 2:
                key = False
                s = i
        i += 1
    return s, summ

s, summ = medinterv(amou)

# Выборочная медиана
vmedi = b[s] + h * ((n / 2 - summ) / amou[s])

# Критерий Колмогорова
# Контрольная функция распр
frasp = [scipy.stats.norm.cdf((i - sred) / otkl) for i in mid]
# Дельты
delta = [fabs(i - j) for i, j in zip(c, frasp)]
dn = max(delta) * (pow(n, 1/2) - 0.01 + (0.85/pow(10, 1/2)))

#Критерий значимости
alfa = 0.1
#Критические значения
critic = {0.15: 0.775, 0.10: 0.819, 0.05: 0.895, 0.03: 0.955, 0.01: 1.035}

# Сетка на графике
figure = pylab.figure(1)
axes = figure.add_subplot(1, 1, 1)

# Оформление графиков
pylab.figure(1)
pylab.bar(b, freque, width=h, color='#3F88C5', edgecolor="#4A525A")
pylab.plot(mid, freque, linestyle="-",
           marker="o",
           color="#24272B",
           markerfacecolor="#ff22aa",
           lw='2')
pylab.ylim(ymax=max(freque) + 0.03)
pylab.xlim(xmax=mid[-1] + h)

# График эмпир функции распределния
figure = pylab.figure(2)
axes2 = figure.add_subplot(1, 1, 1)
pylab.figure(2)
pylab.plot(mid, frasp, linestyle="-",
           marker="o",
           color="#24272B",
           markerfacecolor="#ff22aa",
           lw='2')

pylab.plot(mid, c, linestyle="",
           marker="", )

# Отрисовка стрелок
arrowprops = {'arrowstyle': 'simple', }
for i, j in zip(c, mid):
    pylab.annotate(u'',
                   xy=(j, i),
                   xytext=(j + h, i),
                   arrowprops=arrowprops)
pylab.ylim(ymax=1.1)
pylab.xlim(xmax=mid[-1] + h)

locator = matplotlib.ticker.IndexLocator(h, 0)
locator2 = matplotlib.ticker.IndexLocator(h, 0)
axes.xaxis.set_major_locator(locator)
axes2.xaxis.set_major_locator(locator2)

axes.grid()
axes2.grid()

if dn < critic[alfa]:
    print('Выборка нормально распределена!')
else:
    print('Условие нормальности не выполняется!')
print()
print('D: ' + str(dn))
print('D*: ' + str(critic[alfa]))
print('Среднее: ' + str(sred))
print('Выбороч дисперсия: ' + str(disp))
print('Отклонение: ' + str(otkl))
print('Коэф вариации: ' + str(v))
print('Оценка коэф асимметрии: ' + str(sk))
print('Оценка эксцесса: ' + str(ex))
print('Выборочная мода: ' + str(vmoda))
print('Выборочная медиана: ' + str(vmedi))
print()
print('Интеравалы: ' + str(interv))
print('Частоты: ' + str(amou))
print('Частотность: ' + str(freque))
print('Середины: ' + str(mid))
print('Эмпир функция распр: ' + str(c))
print('Контрольная функция распр: ' + str(frasp))
pylab.show()

f.close()
