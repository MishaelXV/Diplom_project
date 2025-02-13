# Аналитическая модель для интерпретации термометрии нагнетательных скважин

## Описание проекта

Данный проект представляет собой разработку аналитической модели для интерпретации данных термометрии нагнетательных скважин, позволяющей определять параметры пласта без необходимости остановки скважины. Модель основана на решении уравнения теплопроводности в стволе нагнетательной скважины и учитывает динамику теплопереноса в системе скважина-пласт.

Основной целью проекта является создание эффективного инструмента для анализа тепловых профилей в скважинах. Это достигается путем:

1.  **Разработки аналитического решения прямой задачи термометрии:** Модель позволяет рассчитывать температурный профиль в скважине при известных параметрах пласта и режиме закачки.
2.  **Решения обратной задачи термометрии:** Использование аналитического решения прямой задачи для определения неизвестных параметров пласта по измеренным температурным данным.
3.  **Реализации численных методов для решения обратной задачи:** Применение различных методов оптимизации для нахождения оптимального набора параметров пласта, которые наилучшим образом соответствуют измеренным данным.
4.  Проверка модели на тестовых данных.
5.  **Оценки устойчивости и чувствительности модели:** Исследование влияния погрешностей измерений и параметров модели на точность результатов.
6.  **Построения карты применимости модели:** Определение границ применимости модели, основанных на различных параметрах пласта и условиях проведения измерений.

## Методы и подходы

### 1. Аналитическое решение прямой задачи

Для описания температурного распределения в скважине используется аналитическое решение уравнения теплопроводности в цилиндрической системе координат. Учтены основные факторы, влияющие на теплоперенос, такие как:

*   Теплопроводность горных пород и теплоносителя.
*   Теплоемкость горных пород и теплоносителя.
*   Скорость закачки теплоносителя.
*   Температура нагнетания.

### 2. Численное решение обратной задачи

Для решения обратной задачи термометрии использован ряд численных методов оптимизации:

*   Алгоритм Левенберга-Марквардта
*   Алгоритм Нелдера-Мида
*   Алгоритм Пауэлла


### 3. Оценка устойчивости и чувствительности

Проведен анализ устойчивости модели к погрешностям измерений и вариации параметров. Оценена чувствительность результатов решения к изменению входных параметров, что позволяет определить наиболее важные параметры для интерпретации термометрии.

### 4. Карта применимости модели

Для определения областей наиболее эффективного применения модели построена карта применимости, учитывающая диапазон значений параметров пласта.

   


## Области применения

Модель может быть использована для:

*   Оценки проницаемости пласта в окрестности скважины.
*   Определения эффективной теплопроводности пласта.
*   Мониторинга теплового состояния нагнетательных скважин.
*   Оценки эффективности закачки теплоносителя.
*   Анализа распределения температуры и потенциальных зон теплообмена.

## Чтобы начать: 
* Запустить pip install -r requirements.txt в терминале.
* Запустить скрипт app.py.
