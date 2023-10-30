import pickle
import streamlit as st

model = pickle.load(open('estimasi_battery_RUL.sav', 'rb'))

st.title('Estimasi Usia Baterai')

Discharge_Time = st.number_input('Input Waktu Pemakaian Baterai')
Decrement = st.number_input('Input Waktu Pengurangan Baterai Saat 3.6-4.3V')
Max_Voltage_Dischar = st.number_input('Input Pelepasan Tegangan Maksimal')
Min_Voltage_Charg = st.number_input('Input Muatan Tegangan Minimum')
Time = st.number_input('Input Waktu Saat Tegangan 4.15V')
Time_constant_current = st.number_input('Input Arus Waktu Konstan')
Charging_time = st.number_input('Input Waktu Pengisian')

predict = ''

if st.button('Estimasi'):
    predict = model.predict(
        [[Discharge_Time, Decrement, Max_Voltage_Dischar, Min_Voltage_Charg, Time, Time_constant_current, Charging_time]]
    )
    st.write ('Estimasi Sisa Masa Manfaat Baterai : ', predict)