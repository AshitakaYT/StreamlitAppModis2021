import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from PIL import Image
from streamlit_echarts import st_echarts
import json
import plotly.express as px
from Helper import loadData

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- Ingestion
#--------------------------------- ---------------------------------  ---------------------------------

st.set_page_config(page_title='Dashboard ACCIA', page_icon=None, layout='wide', initial_sidebar_state='auto')

machines = loadData('../ACCIA/PdM/Code/PdM_machines.csv')
maint = loadData('../ACCIA/PdM/Code/PdM_maint.csv')
errors = loadData('../ACCIA/PdM/Code/PdM_errors.csv')
telemetry = loadData('../ACCIA/PdM/Code/PdM_telemetry.csv')
failures = loadData('../ACCIA/PdM/Code/PdM_failures.csv')
errorsbymodel = loadData('../ACCIA/PdM/Code/errorsbymodel.csv')
errorsbyage = loadData('../ACCIA/PdM/Code/errorsbyage.csv')
failuresbyage = loadData('../ACCIA/PdM/Code/failuresbyage.csv')
failuresbymodel = loadData('../ACCIA/PdM/Code/failuresbymodel.csv')
sincelastfail = loadData('../ACCIA/PdM/Code/sincelastfail.csv')



selected_metrics = st.sidebar.selectbox(
    label="Choose...", options=['Explications sur les datasets','Dashboard']
    )
if selected_metrics == 'Explications sur les datasets':
    st.write("# EXPLICATIONS SUR LE JEU DE DONNEES")
    st.write("""
        ### Cette app permet l'analyse visuelle du jeu de données de maintenance prédictive d'Azure.
        Avant de présenter les datasets, il est essentiel de comprendre quel sujet ils traitent.
        Le jeu de données analyse un lot de 100 machines telles que :
        """)
    title_image = Image.open("./machine.PNG")
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
            ### Afin d'utiliser le dashboard de manière optimale, choisissez 'Dashboard' dans le menu a gauche, puis fermez celui ci.
            """)
if selected_metrics == 'Dashboard':
    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- 4 columns
    #--------------------------------- ---------------------------------  ---------------------------------
    col1, col2, col3, col4 = st.beta_columns((3, 3, 3, 3))

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Age
    #--------------------------------- ---------------------------------  ---------------------------------
    with col1:
        features = st.selectbox(label="Choisissez une analyse liée a l'age", options=["Répartition de l'age des machines",'Modèles selon les ages','Erreurs selon les ages + moyenne par machine','Failures selon les ages + moyenne par machine'])

        if features == "Répartition de l'age des machines":
            fig = px.bar(machines['age'].value_counts(), width=600, height=400)
            st.plotly_chart(fig)
            fig2 = px.pie(machines, values=machines['age'].value_counts(),names=machines['age'].value_counts().keys(), width=600, height=400)
            st.plotly_chart(fig2)

        if features == 'Modèles selon les ages':			
            fig = px.histogram(machines, x='age' ,
                                color='model',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=600, height=400
            )
            st.plotly_chart(fig)

        if features == 'Failures selon les ages + moyenne par machine':
            df = pd.merge(machines, failures)
            fig = px.histogram(df, x='age' ,
                                color='failure',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=600, height=400
            )
            st.plotly_chart(fig)

            fig2 = px.bar(failuresbyage, x='age', y=[failuresbyage.comp1, failuresbyage.comp2, failuresbyage.comp3, failuresbyage.comp4],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)

        if features == 'Erreurs selon les ages + moyenne par machine':			
            df = pd.merge(errors, machines)
            fig = px.histogram(df, x='age' ,
                                color='errorID',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'},
                                width=600, height=400 # can specify one label per df column)
            )
            st.plotly_chart(fig)
            fig2 = px.bar(errorsbyage, x='age', y=[errorsbyage.error1, errorsbyage.error2, errorsbyage.error3, errorsbyage.error4, errorsbyage.error5],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Model
    #--------------------------------- ---------------------------------  ---------------------------------
    with col2:
        features = st.selectbox(
    	label="Choisissez une analyse liée au modèles", options=["Répartition des modèles",'Modèles selon les ages','Erreurs selon les modèles (moyenne par machine)','Failures selon les modèles (moyenne par machine)']
    	)
        if features == "Répartition des modèles":
            fig = px.bar(machines['model'].value_counts(), width=600, height=400)
            st.plotly_chart(fig)
            fig2 = px.pie(machines, values=machines['model'].value_counts(),names=machines['model'].value_counts().keys(), width=600, height=400)
            st.plotly_chart(fig2)

        if features == 'Modèles selon les ages':
            #Age by model
            fig = px.histogram(machines, x='age' ,
                                color='model',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=600, height=400
            )
            st.plotly_chart(fig)

        if features == 'Failures selon les modèles (moyenne par machine)':
            #Failures by model
            fig2 = px.bar(failuresbymodel, x='model', y=[failuresbymodel.comp1, failuresbymodel.comp2, failuresbymodel.comp3, failuresbymodel.comp4],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)

        if features == 'Erreurs selon les modèles (moyenne par machine)':
            #Errors by model
            fig2 = px.bar(errorsbymodel, x='model', y=[errorsbymodel.error1, errorsbymodel.error2, errorsbymodel.error3, errorsbymodel.error4, errorsbymodel.error5],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Failures
    #--------------------------------- ---------------------------------  ---------------------------------
    with col3:
        features = st.selectbox(
        label="Choisissez une analyse liée aux failures", options=['Répartition des failures','Failures selon les ages + moyenne par machine','Failures selon les modèles + moyenne par machine']
    	)
        if features == 'Répartition des failures':
            fig = px.bar(failures['failure'].value_counts(), width=600, height=400)
            st.plotly_chart(fig)
            fig2 = px.pie(failures, values=failures['failure'].value_counts(),names=failures['failure'].value_counts().keys(), width=600, height=400)
            st.plotly_chart(fig2)

        if features == 'Failures selon les modèles + moyenne par machine':
            fig2 = px.bar(failuresbymodel, x='model', y=[failuresbymodel.comp1, failuresbymodel.comp2, failuresbymodel.comp3, failuresbymodel.comp4],
                        barmode='group',
                        width=500, height=400
                        )
            st.plotly_chart(fig2)

        if features == 'Failures selon les ages + moyenne par machine':
            df = pd.merge(machines, failures)
            fig = px.histogram(df, x='age' ,
                                color='failure',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=600, height=400
            )
            st.plotly_chart(fig)

            fig2 = px.bar(failuresbyage, x='age', y=[failuresbyage.comp1, failuresbyage.comp2, failuresbyage.comp3, failuresbyage.comp4],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)
    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Errors
    #--------------------------------- ---------------------------------  ---------------------------------

    with col4:
        features = st.selectbox(
        label="Choisissez une analyse liée aux erreurs", options=['Répartition des erreurs','Erreurs selon les ages + moyenne par machine','Erreurs selon les modèles + moyenne par machine']
        )
        if features =='Répartition des erreurs':
            fig = px.bar(errors['errorID'].value_counts(), width=600, height=400)
            st.plotly_chart(fig)
            fig2 = px.pie(errors, values=errors['errorID'].value_counts(),names=errors['errorID'].value_counts().keys(), width=600, height=400)
            st.plotly_chart(fig2)
        if features == 'Erreurs selon les modèles + moyenne par machine':
    	    #Errors by model
            fig2 = px.bar(errorsbymodel, x='model', y=[errorsbymodel.error1, errorsbymodel.error2, errorsbymodel.error3, errorsbymodel.error4, errorsbymodel.error5],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)
        if features == 'Erreurs selon les ages + moyenne par machine':
            df = pd.merge(errors, machines)
            fig = px.histogram(df, x='age' ,
                                color='errorID',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=600, height=400
            )
            st.plotly_chart(fig)

            fig2 = px.bar(errorsbyage, x='age', y=[errorsbyage.error1, errorsbyage.error2, errorsbyage.error3, errorsbyage.error4, errorsbyage.error5],
                        barmode='group',
                        width=600, height=400
                        )
            st.plotly_chart(fig2)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- 2 columns
    #--------------------------------- ---------------------------------  ---------------------------------
    lin1, lin2 = st.beta_columns((6, 6))

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- telemetry whole
    #--------------------------------- ---------------------------------  ---------------------------------

    with lin1:
        selected_feature = st.selectbox(
                label="Choisissez un capteur de télémétrie", options=['volt', 'rotate', 'vibration', 'pressure']
                )

        st.write(selected_feature + " pour toutes les machines")
        
        fig = px.histogram(telemetry, x=selected_feature, nbins=1000, width=1200, height=400)
        st.plotly_chart(fig)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- telemetry machine
    #--------------------------------- ---------------------------------  ---------------------------------

    with lin2:
        selected_machine = st.slider("Pick a machine", 1, 100)
        st.write(selected_feature + " pour la machine " + str(selected_machine))
        plot = telemetry[
            telemetry.machineID == selected_machine][["datetime", selected_feature]].set_index("datetime")

        fig = px.line(plot, width=1200, height=400)
        st.plotly_chart(fig)

