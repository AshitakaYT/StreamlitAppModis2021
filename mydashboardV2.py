#import the libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from PIL import Image
from streamlit_echarts import st_echarts
import plotly.express as px
from Helper import loadData
import plotly.graph_objects as go
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pickle
import numpy as np
import streamlit.components.v1 as components

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- Ingestion
#--------------------------------- ---------------------------------  ---------------------------------

st.set_page_config(page_title='Dashboard ACCIA', page_icon=None, layout='wide', initial_sidebar_state='auto')

machines = loadData('./PdM_machines.csv')
machines2 = loadData('./machines2.csv')
maint = loadData('./PdM_maint.csv')
errors = loadData('./PdM_errors.csv')
telemetry = loadData('./PdM_telemetry.csv')
failures = loadData('./PdM_failures.csv')
errorsbymodel = loadData('./errorsbymodel.csv')
errorsbyage = loadData('./errorsbyage.csv')
failuresbyage = loadData('./failuresbyage.csv')
failuresbymodel = loadData('./failuresbymodel.csv')
sincelastfail = loadData('./sincelastfail.csv')
days_between_failures = loadData('./days_between_failures.csv')


selected_metrics = st.sidebar.selectbox(
    label="Choose...", options=['Explications sur les datasets','Etat de santé systeme','Télémétrie','Maintenance']
    )
if selected_metrics == 'Explications sur les datasets':

    st.write("# EXPLICATIONS SUR LE JEU DE DONNEES")
    st.write("""
        Avant de présenter les datasets*, il est essentiel de comprendre quel sujet ils traitent.
        Le jeu de données analyse un lot de 100 machines telles que :
        """)
    title_image = Image.open("./Machine.PNG")
    st.image(title_image)
    st.write("""
            Il existe 4 modèles de machines. Chaque machine est concue a l'aide de 4 pièces (components) qui mesurent diverses données (telemetry). Ces pièces sont maintenues régulièrement, et peuvent générer des erreurs (errors). Lorsqu'une pièce est défaillante, elle est remplacée(failure).
            """)
    st.write("""
            ### Ce jeu de données contient:\n
            -l'historique pour l'année 2015 de toutes les données mesurées chaque heure (telemetry.csv)\n
            -l'historique pour l'année 2015 de toutes les maintenances effectuées (maintenance.csv)\n
            -l'historique pour l'année 2015 de tous les remplacements de pièce effectués (failures.csv)\n
            -l'historique pour l'année 2015 de tous les enregistrements d'erreurs (errors.csv)\n
            -le listing de chaque machine, de leur modèle et de leur age (machines.csv)\n

            *lien de téléchargement : https://github.com/ThereIsNoSpoonMrAnderson/StreamlitAppModis2021
            """)
if selected_metrics == 'Etat de santé systeme':
    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- 4 columns
    #--------------------------------- ---------------------------------  ---------------------------------
    col1, col2 = st.beta_columns((6, 6))
    col3, col4 = st.beta_columns((6, 6))
    width = 1200
    height = 400

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Age
    #--------------------------------- ---------------------------------  ---------------------------------
    with col1:
        features = st.selectbox(label="Choisissez une analyse liée a l'age", options=["Répartition de l'age des machines",'Modèles selon les ages','Erreurs selon les ages + moyenne par machine','Failures selon les ages + moyenne par machine'])

        if features == "Répartition de l'age des machines":
            fig = px.bar(machines['age'].value_counts(), width=width, height=height, labels={'index':'age', 'value': 'nombre de machines'})
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.pie(machines, values=machines['age'].value_counts(),names=machines['age'].value_counts().keys(), width=width, height=height)
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Modèles selon les ages':			
            fig = px.histogram(machines, x='age' ,
                                color='model',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'nombre de machines'}, # can specify one label per df column)
                                width=width, height=height
            )
            st.plotly_chart(fig, use_container_width=True)

        if features == 'Failures selon les ages + moyenne par machine':
            df = pd.merge(machines, failures)
            fig = px.histogram(df, x='age' ,
                                color='failure',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'count':'nombre de défaillances'}, # can specify one label per df column)
                                width=width, height=height
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(failuresbyage, x='age', y=[failuresbyage.comp1, failuresbyage.comp2, failuresbyage.comp3, failuresbyage.comp4],
                        barmode='group',
                        labels={'value':'nombre de défaillances'}, # can specify one label per df column)
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Erreurs selon les ages + moyenne par machine':			
            df = pd.merge(errors, machines)
            fig = px.histogram(df, x='age' ,
                                color='errorID',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age', 'count':"nombre d'erreurs"},
                                width=width, height=height # can specify one label per df column)
            )
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.bar(errorsbyage, x='age', y=[errorsbyage.error1, errorsbyage.error2, errorsbyage.error3, errorsbyage.error4, errorsbyage.error5],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Model
    #--------------------------------- ---------------------------------  ---------------------------------
    with col2:
        features = st.selectbox(
    	label="Choisissez une analyse liée au modèles", options=["Répartition des modèles",'Modèles selon les ages','Erreurs selon les modèles (moyenne par machine)','Failures selon les modèles (moyenne par machine)']
    	)
        if features == "Répartition des modèles":
            fig = px.bar(machines['model'].value_counts(), width=width, height=height, labels={'index':'modèles', 'value': 'nombre de machines'})
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.pie(machines, values=machines['model'].value_counts(),names=machines['model'].value_counts().keys(), width=width, height=height)
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Modèles selon les ages':
            #Age by model
            fig = px.histogram(machines, x='age' ,
                                color='model',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=width, height=height
            )
            st.plotly_chart(fig, use_container_width=True)

        if features == 'Failures selon les modèles (moyenne par machine)':
            #Failures by model
            fig2 = px.bar(failuresbymodel, x='model', y=[failuresbymodel.comp1, failuresbymodel.comp2, failuresbymodel.comp3, failuresbymodel.comp4],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Erreurs selon les modèles (moyenne par machine)':
            #Errors by model
            fig2 = px.bar(errorsbymodel, x='model', y=[errorsbymodel.error1, errorsbymodel.error2, errorsbymodel.error3, errorsbymodel.error4, errorsbymodel.error5],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Failures
    #--------------------------------- ---------------------------------  ---------------------------------
    with col3:
        features = st.selectbox(
        label="Choisissez une analyse liée aux failures", options=['Répartition des failures','Failures selon les ages + moyenne par machine','Failures selon les modèles + moyenne par machine']
    	)
        if features == 'Répartition des failures':
            fig = px.bar(failures['failure'].value_counts(), width=width, height=height, labels={'index':'failures', 'value': 'nombre de machines'})
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.pie(failures, values=failures['failure'].value_counts(),names=failures['failure'].value_counts().keys(), width=width, height=height)
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Failures selon les modèles + moyenne par machine':
            fig2 = px.bar(failuresbymodel, x='model', y=[failuresbymodel.comp1, failuresbymodel.comp2, failuresbymodel.comp3, failuresbymodel.comp4],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Failures selon les ages + moyenne par machine':
            df = pd.merge(machines, failures)
            fig = px.histogram(df, x='age' ,
                                color='failure',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=width, height=height
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(failuresbyage, x='age', y=[failuresbyage.comp1, failuresbyage.comp2, failuresbyage.comp3, failuresbyage.comp4],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)
    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Errors
    #--------------------------------- ---------------------------------  ---------------------------------

    with col4:
        features = st.selectbox(
        label="Choisissez une analyse liée aux erreurs", options=['Répartition des erreurs','Erreurs selon les ages + moyenne par machine','Erreurs selon les modèles + moyenne par machine']
        )
        if features =='Répartition des erreurs':
            fig = px.bar(errors['errorID'].value_counts(), width=width, height=height, labels={'index':'erreurs', 'value': 'nombre de machines'})
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.pie(errors, values=errors['errorID'].value_counts(),names=errors['errorID'].value_counts().keys(), width=width, height=height)
            st.plotly_chart(fig2, use_container_width=True)
        if features == 'Erreurs selon les modèles + moyenne par machine':
    	    #Errors by model
            fig2 = px.bar(errorsbymodel, x='model', y=[errorsbymodel.error1, errorsbymodel.error2, errorsbymodel.error3, errorsbymodel.error4, errorsbymodel.error5],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)
        if features == 'Erreurs selon les ages + moyenne par machine':
            df = pd.merge(errors, machines)
            fig = px.histogram(df, x='age' ,
                                color='errorID',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=width, height=height
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(errorsbyage, x='age', y=[errorsbyage.error1, errorsbyage.error2, errorsbyage.error3, errorsbyage.error4, errorsbyage.error5],
                        barmode='group',
                        width=width, height=height
                        )
            st.plotly_chart(fig2, use_container_width=True)


    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- telemetry whole
    #--------------------------------- ---------------------------------  ---------------------------------
if selected_metrics == 'Télémétrie':
    lin1, lin2 = st.beta_columns((6, 6))
    width = 1200
    height = 400
    times = ["1 jour", "1 semaine", "1 mois"]


    with lin1:
        time = st.radio("Choisissez une analyse de voltage", times)

        if time == '1 mois':

            selected_machine1 = st.slider("Choisissez une machine pour le voltage", 1, 100)
            st.write('volt' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'volt']].set_index("datetime").tail(25 * 7 * 4)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)


        elif time == '1 semaine':    
            selected_machine1 = st.slider("Choisissez une machine pour le voltage", 1, 100)
            st.write('volt' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'volt']].set_index("datetime").tail(25 * 7)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

        elif time == '1 jour':    
            selected_machine1 = st.slider("Choisissez une machine pour le voltage", 1, 100)
            st.write('volt' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'volt']].set_index("datetime").tail(25)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

        time= st.radio("Choisissez une analyse de rotation", times)

        if time == '1 mois':

            selected_machine1 = st.slider("Choisissez une machine pour la  rotation", 1, 100)
            st.write('rotate' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'rotate']].set_index("datetime").tail(25 * 7 * 4)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)


        elif time == '1 semaine':    
            selected_machine1 = st.slider("Choisissez une machine pour la rotation", 1, 100)
            st.write('rotate' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'rotate']].set_index("datetime").tail(25 * 7)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

        elif time == '1 jour':    
            selected_machine1 = st.slider("Choisissez une machine pour la rotation", 1, 100)
            st.write('rotate' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'rotate']].set_index("datetime").tail(25)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- telemetry machine
    #--------------------------------- ---------------------------------  ---------------------------------

    with lin2:

        time = st.radio("Choisissez une analyse de vibration", times)

        if time == '1 mois':

            selected_machine1 = st.slider("Choisissez une machine pour la vibration", 1, 100)
            st.write('vibration' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'vibration']].set_index("datetime").tail(25 * 7 * 4)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)


        elif time == '1 semaine':    
            selected_machine1 = st.slider("Choisissez une machine pour le vibration", 1, 100)
            st.write('vibration' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'vibration']].set_index("datetime").tail(25 * 7)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

        elif time == '1 jour':    
            selected_machine1 = st.slider("Choisissez une machine pour le vibration", 1, 100)
            st.write('vibration' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'vibration']].set_index("datetime").tail(25)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

        time= st.radio("Choisissez une analyse de pression", times)

        if time == '1 mois':

            selected_machine1 = st.slider("Choisissez une machine pour la  pression", 1, 100)
            st.write('pressure' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'pressure']].set_index("datetime").tail(25 * 7 * 4)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)


        elif time == '1 semaine':    
            selected_machine1 = st.slider("Choisissez une machine pour la pression", 1, 100)
            st.write('pressure' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'pressure']].set_index("datetime").tail(25 * 7)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)

        elif time == '1 jour':    
            selected_machine1 = st.slider("Choisissez une machine pour la pression", 1, 100)
            st.write('pressure' + " pour la machine " + str(selected_machine1))
            plot = telemetry[
                telemetry.machineID == selected_machine1][["datetime", 'pressure']].set_index("datetime").tail(25)

            fig = px.line(plot, width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)


if selected_metrics == 'Maintenance':

    model_comp1 = pickle.load(open('comp1_pred_model.pkl', 'rb'))
    model_comp2 = pickle.load(open('comp2_pred_model.pkl', 'rb'))
    model_comp3 = pickle.load(open('comp3_pred_model.pkl', 'rb'))
    model_comp4 = pickle.load(open('comp4_pred_model.pkl', 'rb'))

    def get_days(df, kind, failure, machine):
        first_time = df['datetime'].loc[(df[kind] == failure) & (df.machineID == machine)]
        if len(first_time) == 0:
            return(365)
        first_time = first_time.iloc[len(first_time) - 1]

        later_time = df['datetime'].iloc[len(df['datetime']) - 1]

        first_times = datetime.datetime.strptime(first_time,'%Y-%m-%d %H:%M:%S')
        later_times = datetime.datetime.strptime(later_time,'%Y-%m-%d %H:%M:%S')

        difference = later_times - first_times

        seconds_in_day = 24 * 60 * 60

        duration_in_s = difference.total_seconds()
        days  = difference.days                         # Build-in datetime function
        days  = divmod(duration_in_s, 86400)[0]
        return(days)

    dayssincelastfailure = pd.merge(telemetry, failures, how = 'outer')
    dayssincelastmaint = pd.merge(telemetry, maint, how= 'outer')
    maint1, maint2 = st.beta_columns((6, 6))
    width = 1200
    height = 400

    with maint1:

        selected_machine1 = st.slider("Choisissez une machine (défaillance)", 1, 100)
        st.write('JOURS DEPUIS LA DERNIERE DEFAILLANCE DES PIECES POUR LA MACHINE ' + str(selected_machine1))


    with maint2:

        selected_machine2 = st.slider("Choisissez une machine (maintenance)", 1, 100)
        st.write('JOURS DEPUIS LA DERNIERE MAINTENANCE DES PIECES POUR LA MACHINE ' + str(selected_machine2))

    maintquad1, maintquad2, maintquad3, maintquad4 = st.beta_columns((3, 3, 3, 3))
    with maintquad1:

        ycomp1 = np.array([selected_machine1,
            telemetry[telemetry.machineID == selected_machine1]['volt'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['rotate'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['pressure'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['vibration'].tail(1),
            get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1),
            machines2.loc[(machines2['machineID'] == selected_machine1)].model_label,
            machines2.loc[(machines2['machineID'] == selected_machine1)].age
       ])
        ycomp1 = ycomp1.reshape(1, -1)
        predictioncomp1 = model_comp1.predict(ycomp1)

        ycomp2 = np.array([selected_machine1,
            telemetry[telemetry.machineID == selected_machine1]['volt'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['rotate'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['pressure'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['vibration'].tail(1),
            get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1),
            machines2.loc[(machines2['machineID'] == selected_machine1)].model_label,
            machines2.loc[(machines2['machineID'] == selected_machine1)].age
       ])
        ycomp2 = ycomp2.reshape(1, -1)
        predictioncomp2 = model_comp2.predict(ycomp2)

        if predictioncomp1 == 0:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 1", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 500], 'color': 'white'}]
                        }))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        elif predictioncomp1 == 1:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 1", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 1.25, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 1.2 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 1.2, 500], 'color': 'red'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 1.25 }}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        else:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 1", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 1.1, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 0.95 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) *0.95, 500], 'color': 'orange'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp1', selected_machine1) * 1.1}}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        if predictioncomp2 == 0:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 2", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 500], 'color': 'white'}]
                        }))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        elif predictioncomp2 == 1:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 2", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 1.25, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 1.2 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 1.2, 500], 'color': 'red'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 1.25 }}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        else:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 2", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 1.1, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 0.95 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) *0.95, 500], 'color': 'orange'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp2', selected_machine1) * 1.1}}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

    with maintquad2:



        ycomp3 = np.array([selected_machine1,
            telemetry[telemetry.machineID == selected_machine1]['volt'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['rotate'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['pressure'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['vibration'].tail(1),
            get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1),
            machines2.loc[(machines2['machineID'] == selected_machine1)].model_label,
            machines2.loc[(machines2['machineID'] == selected_machine1)].age
       ])
        ycomp3 = ycomp3.reshape(1, -1)
        predictioncomp3 = model_comp3.predict(ycomp3)


        ycomp4 = np.array([selected_machine1,
            telemetry[telemetry.machineID == selected_machine1]['volt'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['rotate'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['pressure'].tail(1),
            telemetry[telemetry.machineID == selected_machine1]['vibration'].tail(1),
            get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1),
            machines2.loc[(machines2['machineID'] == selected_machine1)].model_label,
            machines2.loc[(machines2['machineID'] == selected_machine1)].age
       ])
        ycomp4 = ycomp4.reshape(1, -1)
        predictioncomp4 = model_comp4.predict(ycomp4)




        if predictioncomp3 == 0:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 3", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 500], 'color': 'white'}]
                        }))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        elif predictioncomp3 == 1:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 3", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 1.25, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 1.2 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 1.2, 500], 'color': 'red'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 1.25 }}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        else:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 3", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 1.1, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 0.95 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) *0.95, 500], 'color': 'orange'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp3', selected_machine1) * 1.1}}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)


        if predictioncomp4 == 0:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 4", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 500], 'color': 'white'}]
                        }))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        elif predictioncomp4 == 1:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 4", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 1.25, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 1.2 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 1.2, 500], 'color': 'red'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 1.25 }}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

        else:

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Pièce 4", 'font': {'size': 20}},
                delta = {'reference': get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 1.1, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 0.8 ], 'color': 'green'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 0.8, get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 0.95 ], 'color': 'yellow'},
                        {'range': [get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) *0.95, 500], 'color': 'orange'},],
                    'threshold': {
                        'line': {'color': "white", 'width': 1},
                        'thickness': 0.75,
                        'value': get_days(dayssincelastfailure,'failure', 'comp4', selected_machine1) * 1.1}}))

            fig.update_layout(autosize = False, height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

            st.plotly_chart(fig, use_container_width=True)

    with maintquad3:
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = get_days(dayssincelastmaint,'comp', 'comp1', selected_machine2),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pièce 1", 'font': {'size': 20}},
            delta = {'reference': 180, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 250], 'color': 'white'},
                    {'range': [180, 500], 'color': 'orange'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 180}}))

        fig.update_layout(height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = get_days(dayssincelastmaint,'comp', 'comp3', selected_machine2),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pièce 2", 'font': {'size': 20}},
            delta = {'reference': 180, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 250], 'color': 'white'},
                    {'range': [180, 500], 'color': 'orange'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 180}}))

        fig.update_layout(height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

        st.plotly_chart(fig, use_container_width=True)

    with maintquad4:
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = get_days(dayssincelastmaint,'comp', 'comp3', selected_machine2),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pièce 3", 'font': {'size': 20}},
            delta = {'reference': 180, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 250], 'color': 'white'},
                    {'range': [180, 500], 'color': 'orange'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 180}}))

        fig.update_layout(height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = get_days(dayssincelastmaint,'comp', 'comp4', selected_machine2),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pièce 4", 'font': {'size': 20}},
            delta = {'reference': 180, 'increasing': {'color': "red"}, 'decreasing' : {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 250], 'color': 'white'},
                    {'range': [180, 500], 'color': 'orange'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 180}}))

        fig.update_layout(height=250, paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

        st.plotly_chart(fig, use_container_width=True)


    maint1, maint2 = st.beta_columns((6, 6))
    width = 1200
    height = 400

    with maint1:
        st.write('HISTORIQUE DES DERNIERES DEFAILLANCE DES PIECES POUR LA MACHINE ' + str(selected_machine1))
        plot = failures[
            failures.machineID == selected_machine1][["datetime", 'failure']].set_index("datetime")

        fig = px.scatter(plot, width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

    with maint2:
        st.write('HISTORIQUE DES DERNIERES MAINTENANCES DES PIECES POUR LA MACHINE ' + str(selected_machine2))
        plot = maint[
            maint.machineID == selected_machine1][["datetime", 'comp']].set_index("datetime")

        fig = px.scatter(plot, width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)
