import joblib
import streamlit as st
import numpy as np
import sklearn
print(sklearn.__version__)

from input_processing import preprocess


# Функция для предсказания на основе входных данных
def predict(features):
    model = joblib.load("credit_risk_model1.joblib")
    return model.predict(preprocess(features))

#Заголовок для приложения
st.title("Кредитный скоринг")

# Ввод параметров от пользователя
#age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
Unnamed = st.number_input("Номер", min_value=1, max_value=1000, value=1)
Age = st.number_input("Возраст", min_value=19, max_value=75, value=19)
Sex = st.selectbox(
    "Пол",
    ("male", "female")
)
Job = st.number_input("Работа", min_value=0, max_value=3, value=1)
Housing = st.selectbox("Недвижимость", ('own', 'free', 'rent'))
Saving_accounts = st.selectbox("Сберегательные счета", ('nan', 'little', 'quite rich', 'rich', 'moderate'))
Checking_account = st.selectbox("Текущий счёт", ('little', 'moderate', 'nan', 'rich'))
Duration = st.number_input("Длительность", min_value=4, max_value=72, value=4)
Purpose = st.selectbox("Цель", ('radio/TV', 'education', 'furniture/equipment', 'car', 'business',
       'domestic appliances', 'repairs', 'vacation/others'))

if not (19 <= Age <= 75):
    st.error("Возраст должен быть в диапазоне от 19 до 75 лет")

if not (0 <= Job <= 3):
    st.error("Кол-во работ должно быть неотрицательным и не более 3-х")

if not (4 <= Duration <= 72):
        st.error("Длительность должна быть между 4 и 72")


# Добавьте сюда другие необходимые параметры

# Сбор всех параметров в один массив
features = np.array([Unnamed, Age, Sex, Job, Housing, Saving_accounts,
       Checking_account, Duration, Purpose])

# Кнопка для выполнения предсказания
if st.button("Проверить кредитоспособность"):
    result = predict(features)
    if result > 0.5:  # Предположим, что результат 0.5 - это порог
        st.success("Кредит одобрен!")
    else:
        st.error("Кредит не одобрен")