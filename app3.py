import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt
#-----------------------------------
# Funciones de preprocesamiento 
#-----------------------------------
def encoding_labels(df):
    df = df.reset_index(drop=True)

    # Codificación One-Hot para la columna «Stage»
    enc_stage = OneHotEncoder(sparse=False)
    transformed_data = enc_stage.fit_transform(df[ohe_stage])
    transformed_df = pd.DataFrame(
        transformed_data,
        columns=enc_stage.get_feature_names_out(),
        index=df.index
    )

    # Concatenar el DataFrame original (sin las columnas OHE) con el nuevo DataFrame
    transformed_df = pd.concat([df.drop(ohe_stage, axis=1), transformed_df], axis=1)

    # Codificación One-Hot para la columna «Event type»
    enc = OneHotEncoder(sparse=False)
    transformed_data = enc.fit_transform(transformed_df[ohe_event])

    transformed_event_df = pd.DataFrame(
        transformed_data,
        columns=enc.get_feature_names_out(),
        index=transformed_df.index
    )

    # Concatenar el DataFrame original (sin las columnas OHE) con el nuevo DataFrame OHE
    transformed_df = pd.concat([transformed_df.drop(ohe_event, axis=1), transformed_event_df], axis=1)

    # Concatenar las columnas del OHE faltantes
    cols_to_use = df_ohe.columns.difference(transformed_df.columns)
    transformed_df = pd.merge(transformed_df, df_ohe[cols_to_use], left_index=True, right_index=True)

    # Convertir el DataFrame a un diccionario
    dict_arrays = {k: v.to_numpy() for k, v in transformed_df.items()}

    return transformed_df, dict_arrays



def split_sequences(dict_arrays, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(dict_arrays['Stage_W'])):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(dict_arrays['Stage_W']):
            break

        column_input = dict_arrays.keys()

        seq_x = []
        for column in column_input:
          seq_x.append(dict_arrays[column][i:end_ix])

        seq_y = []
        for i in ["Stage_W", "Stage_N1", "Stage_N2", "Stage_N3", "Stage_R"]:
          seq_y.append(dict_arrays[i][end_ix:out_end_ix])

        X.append(np.transpose(np.array(seq_x))) 
        y.append(np.transpose(np.array(seq_y)))
    return np.array(X), np.array(y)
#------------------------------------------------
#------------------------------------------------
# Funcion de la grafica 
def plot_predictions_patient ( y_test, seq, X_train):

    y_pred = model.predict(X_train)

    max_test = np.argmax(y_test, axis=2)
    max_pred = np.argmax(y_pred, axis=2)
    max_x = np.argmax(X_train[:,:,6:11], axis=2)

    i = 0

    ax1 = plt.figure(figsize=(15, 5))

    # Graficar las etiquetas reales y las predicciones
    plt.plot(np.arange(0,30,1), max_x[i], 's-', label='Actual', color = 'slategrey')
    plt.plot(np.arange(29,31,1), [max_x[i][29], max_test[i][0]], 's--', color = 'lightseagreen')
    plt.plot(np.arange(30,35,1), max_test[i], 's--', label='Real', color = 'lightseagreen')
    plt.plot(np.arange(29,31,1), [max_x[i][29], max_pred[i][0]],'s--',  color = 'mediumpurple')
    plt.plot(np.arange(30,35,1), max_pred[i], 's--', label = 'Predicción', color = 'mediumpurple')

    plt.xticks(ticks=np.arange(35), labels=np.arange(1, 35+ 1))
    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.legend()
    st.pyplot()

#--------------------------------------------------------------------------------------------


# Cargar el modelo
model = tf.keras.models.load_model("modelo900.h5")
label_mapping = {0: "Stage_W", 1: "Stage_N1", 2: "Stage_N2", 3: "Stage_N3", 4: "Stage_R"}
st.title("Predicción de las etapas del sueño")

# Agregar un campo de carga de archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file, index_col=False)
    df= df.drop("Unnamed: 0", axis=1)

    # Mostrar las primeras filas del DataFrame 
    st.write("Primeras filas del archivo CSV:")
    st.write(df.head())

    # Agregar un campo de entrada de tiempo para especificar desde cuándo se quiere analizar
    start_time_input = st.text_input("Ingresa la fila de inicio:", "1")
    start_time = int(start_time_input )

    # Filtrar el DataFrame para incluir solo las filas desde el tiempo de inicio hasta los siguientes 30 minutos
    end_time = start_time + 35
    filtered_df = df.iloc[start_time: end_time]

    #filtered_df['Time'] = pd.to_datetime(filtered_df['Time'], format='%H:%M:%S')
    input_dataframes=[filtered_df]

    if st.button("Predecir"):
        
        events_list = []
        for df in input_dataframes:
            for i in df["Event type"].unique():
                if not i in events_list:
                    events_list.append(i)
        events_list = ["Event type_" + event_type for event_type in events_list]
        events_list = events_list + ["Stage_W", "Stage_N1", "Stage_N2", "Stage_N3", "Stage_R"]
        data = ["Stage_W", "Stage_N1", "Stage_N2", "Stage_N3", "Stage_R"]
        
        # De acuerdo con la cantidad de eventos, se inicializa un DF para aplicar un One-Hot Encoder
        df_ohe = pd.read_csv('df_one.csv') 
        df_ohe= df_ohe.drop("Unnamed: 0", axis=1)

        # Emplear un OHE distinto para cada una de las dos columnas categóricas
        ohe_stage = ["Stage"]
        ohe_event = ["Event type"]

        train_dataframes_sample= input_dataframes
        # Aplicar la función de transformación de las categorías a cada DataFrame de entrenamiento
        tuples_array = [encoding_labels(df) for df in train_dataframes_sample]
        
        train_transformed_dataframes = [i[0] for i in tuples_array]
        train_dictionaries = [i[1] for i in tuples_array]        

        train_scaled = train_dictionaries
        # Escoger el número de entradas y salidas
        n_steps_in = 1*30      # Considerar los 30 minutos previos
        n_steps_out = 5      # Predecir los siguientes 5 minutos
        n_features = 986       # Columnas resultantes

        dict_train = train_dictionaries[0]
        X_train, Y_train= split_sequences(dict_train, n_steps_in, n_steps_out)
    
        # Realizar la predicción con el modelo
        # X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_pred = model.predict(X_train)
        y_pred_classes = np.argmax(y_pred, axis=-1)
        
        
        # Convertir los índices de las etiquetas a etiquetas originales utilizando el diccionario
        st.write("Resultado de la predicción para los próximos 5 minutos:")
        #st.write(y_pred_classes)


        resp = [data[i] for i in y_pred_classes[0]]
        st.write("Minuto 1: " + resp[0])
        st.write("Minuto 2: " + resp[1])
        st.write("Minuto 3: " + resp[2])
        st.write("Minuto 4: " + resp[3])
        st.write("Minuto 5: " + resp[4])
        
        plot_predictions_patient(Y_train, start_time , X_train  )
