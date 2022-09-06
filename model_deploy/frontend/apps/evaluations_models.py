from doctest import DocFileTest
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    st.title('Avaliação dos modelos')

    st.write("Performance do modelo RandomForest para o problema de Diabetes.")

    st.markdown("### Desempenho")

    df = pd.DataFrame({
        'date': ['2021-01-10','2021-01-11', '2021-01-12', '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06'],
        'RF_V1': [0.85, 0.81, 0.76, 0.68, np.NaN , np.NaN , np.NaN , np.NaN , np.NaN],
        'RF_V2': [np.NaN , np.NaN , np.NaN , 0.94, 0.81, 0.62, np.NaN , np.NaN , np.NaN],
        'RF_V3': [np.NaN , np.NaN , np.NaN , np.NaN, np.NaN, 0.98, 0.90, 0.85 , 0.75],
        'Limiar': [0.70 , 0.70 , 0.70 , 0.70 , 0.70 , 0.70 , 0.70 , 0.70 , 0.70]
    })

    df = df.set_index('date')
    st.line_chart(df)