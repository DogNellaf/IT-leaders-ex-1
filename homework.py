from pandas import DataFrame
from pandas.errors import ParserError
from scipy.stats import stats

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


DEBUG_MODE = True
DEFAULT_PATH = "test.csv"
DEFAULT_ALPHA = 0.05


@st.cache_data
def load_dataset(path: str) -> DataFrame | None:
    try:
        dataset = pd.read_csv(path)
    except FileNotFoundError:
        st.write('Не удалось загрузить датасет')
        st.write('Файл не найден')
        return None
    except ParserError as e:
        st.write('Не удалось загрузить датасет')
        st.write(f'Ошибка при считывании CSV: {e}')
        return None
    except Exception as e:
        st.write('Не удалось загрузить датасет')
        st.write(f'Получено исключение: {e}')
        return None

    if dataset is None or dataset.empty:
        st.write('Не удалось загрузить датасет')
        st.write(f'Необработанная ошибка')
        return None

    st.write('Датасет успешно загружен')
    return dataset


def load_options(dataset: DataFrame) -> list:
    options = st.multiselect(
        'Столбцы для анализа:',
        dataset.columns,
        max_selections=2)

    if DEBUG_MODE:
        st.write('[DEBUG] Были выбраны следующие параметры:', options)

    return options


def make_plots(dataset, first, second):

    if DEBUG_MODE:
        st.write("[DEBUG] Типы данных в датасете:", dataset.dtypes)

    categorical = dataset.select_dtypes(include=['object']).columns.tolist()

    if DEBUG_MODE:
        st.write("[DEBUG] Какие типы являются категориальными:", categorical)

    st.write(f"Распределение столбцов {first} и {second}")

    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    bins = st.slider("Количество делений", 1, 100, 15)

    if first in categorical:
        first_sum = dataset[first].value_counts()

        if DEBUG_MODE:
            st.write("[DEBUG] Полученные данные перед построением диаграммы:", first_sum)

        axs[0].pie(first_sum, labels=first_sum.index)
    else:
        axs[0].hist(dataset[first], bins=bins)

    if second in categorical:
        second_sum = dataset[second].value_counts()

        if DEBUG_MODE:
            st.write("[DEBUG] Полученные данные перед построением диаграммы:", second_sum)

        axs[0].pie(second_sum, labels=second_sum.index)
    else:
        axs[1].hist(dataset[second], bins=bins)

    st.pyplot(fig)

def load_hypothesis() -> str:
    hypothesis = st.selectbox(
        'Гипотеза:',
        ['Один образец t-критерия', 'Два выборочных t-теста'])

    if DEBUG_MODE:
        st.write('[DEBUG] Была выбрана следующая гипотеза:', hypothesis)

    return hypothesis

def analysis_hypothesis(dataset, hypothesis, first, second):

    categorical = dataset.select_dtypes(include=['object']).columns.tolist()

    if DEBUG_MODE:
        st.write("[DEBUG] Какие типы являются категориальными:", categorical)

    if hypothesis == "Два выборочных t-теста":
        if first in categorical or second in categorical:
            st.write('Один из столбцов является категориальным')
            return

        first_var = np.var(dataset[first])
        second_var = np.var(dataset[second])

        st.write('Дисперсия первого столбца:', first_var)
        st.write('Дисперсия второго столбца:', second_var)

        if abs(first_var - second_var) >= 4:
            st.write('Столбцы не имеют равную дисперсию')
            return

        result = stats.ttest_ind(a=dataset[first], b=dataset[second], equal_var=True)

        t = result.statistic
        p = result.pvalue

        st.write(f'Статистика t-теста = {t} , '
                 f'соответствующее двустороннее значение p = {p}.', )

        if p > DEFAULT_ALPHA:
            st.write(f"Поскольку p-значение нашего теста ({p}) больше, чем альфа = {DEFAULT_ALPHA}, "
                  f"у нас нет достаточных доказательств, что среднеее значение у столбцов {first} и {second} отличается")
        else:
            st.write(f"Поскольку p-значение нашего теста ({p}) меньше или равно альфа = {DEFAULT_ALPHA}, "
                  f"мы можем сказать, что среднеее значение столбцов {first} и {second} отличается")

        if DEBUG_MODE:
            st.write('[DEBUG] Реальное среднее значение первого столбца:', dataset[first].mean())
            st.write('[DEBUG] Реальное среднее значение второго столбца:', dataset[second].mean())

        return


    choose = st.selectbox('Столбец для проверки гипотезы:', [first, second])

    if DEBUG_MODE:
        st.write('[DEBUG] Был выбран следующий столбец:', choose)

    if choose in categorical:
        st.write('Выбран категориальный столбец:', choose)
        return

    if hypothesis == "Один образец t-критерия":

        value = st.text_input("Введите проверяемое значение для гипотезы:")

        if value == "":
            st.write("Для продолжения нужно ввести значение")
            return

        try:
            value = float(value)
        except:
            st.write("Введенное значение нельзя сконвертировать в число")
            return

        values = list(dataset[choose])

        if DEBUG_MODE:
            st.write('[DEBUG] Значения столбца:', values)

        result = stats.ttest_1samp(a=values, popmean=value)
        t = result.statistic
        p = result.pvalue
        st.write(f'Статистика t-теста = {t} , '
                 f'соответствующее двустороннее значение p = {p}.', )

        if p > DEFAULT_ALPHA:
            st.write(f"Поскольку p-значение нашего теста ({p}) больше, чем альфа = {DEFAULT_ALPHA}, "
                  f"у нас нет достаточных доказательств, что среднеее значение отличается от {value}")
        else:
            st.write(f"Поскольку p-значение нашего теста ({p}) меньше или равно альфа = {DEFAULT_ALPHA}, "
                  f"мы можем сказать, что среднеее значение отличается от {value}")

        if DEBUG_MODE:
            st.write('[DEBUG] Реальное среднее значение столбца:', dataset[choose].mean())



def main():
    global DEBUG_MODE
    st.write('Выполнил Сидоров Данил')
    DEBUG_MODE = st.checkbox("Режим отладки")
    st.title('Программа для промежуточной аттестации')
    path = st.text_input("Введите путь до датасета:", value=DEFAULT_PATH if DEBUG_MODE else "")
    dataset = load_dataset(path)

    if dataset is None:
        return

    options = load_options(dataset)

    if len(options) != 2:
        st.write('Нужно обязательно выбрать два столбца')
        return

    first = options[0]
    second = options[1]

    make_plots(dataset, first, second)

    hypothesis = load_hypothesis()

    analysis_hypothesis(dataset, hypothesis, first, second)


if __name__ == '__main__':
    main()
