import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from PIL import Image
from streamlit_echarts import st_echarts
import plotly.express as px
from Helper import loadData

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- Ingestion
#--------------------------------- ---------------------------------  ---------------------------------

st.set_page_config(page_title='Dashboard ACCIA', page_icon=None, layout='wide', initial_sidebar_state='auto')

machines = loadData('./PdM_machines.csv')
maint = loadData('./PdM_maint.csv')
errors = loadData('./PdM_errors.csv')
telemetry = loadData('./PdM_telemetry.csv')
failures = loadData('./PdM_failures.csv')
errorsbymodel = loadData('./errorsbymodel.csv')
errorsbyage = loadData('./errorsbyage.csv')
failuresbyage = loadData('./failuresbyage.csv')
failuresbymodel = loadData('./failuresbymodel.csv')
sincelastfail = loadData('./sincelastfail.csv')



selected_metrics = st.sidebar.selectbox(
    label="Choose...", options=['Explications sur les datasets','Analyse de données','Etat de santé systeme']
    )
if selected_metrics == 'Explications sur les datasets':
    st.write("# EXPLICATIONS SUR LE JEU DE DONNEES")
    st.write("""
        ### Cette app permet l'analyse visuelle du jeu de données de maintenance prédictive d'Azure.
        Avant de présenter les datasets, il est essentiel de comprendre quel sujet ils traitent.
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
            ### Afin d'utiliser le dashboard de manière optimale, choisissez 'Dashboard' dans le menu a gauche, puis fermez celui ci.
            """)
if selected_metrics == 'Analyse de données':
    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- 4 columns
    #--------------------------------- ---------------------------------  ---------------------------------
    col1, col2 = st.beta_columns((6, 6))
    col3, col4 = st.beta_columns((6, 6))
    width = 1200
    height = 450

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- Age
    #--------------------------------- ---------------------------------  ---------------------------------
    with col1:
        features = st.selectbox(label="Choisissez une analyse liée a l'age", options=["Répartition de l'age des machines",'Modèles selon les ages','Erreurs selon les ages + moyenne par machine','Failures selon les ages + moyenne par machine'])

        if features == "Répartition de l'age des machines":
            fig = px.bar(machines['age'].value_counts(), width=width, height=height)
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.pie(machines, values=machines['age'].value_counts(),names=machines['age'].value_counts().keys(), width=width, height=height)
            st.plotly_chart(fig2, use_container_width=True)

        if features == 'Modèles selon les ages':			
            fig = px.histogram(machines, x='age' ,
                                color='model',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'}, # can specify one label per df column)
                                width=width, height=height
            )
            st.plotly_chart(fig, use_container_width=True)

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

        if features == 'Erreurs selon les ages + moyenne par machine':			
            df = pd.merge(errors, machines)
            fig = px.histogram(df, x='age' ,
                                color='errorID',
                                #log_y=True, # represent bars with log scale
                                nbins= 50,
                                labels={'value':'age'},
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
            fig = px.bar(machines['model'].value_counts(), width=width, height=height)
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
            fig = px.bar(failures['failure'].value_counts(), width=width, height=height)
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
            fig = px.bar(errors['errorID'].value_counts(), width=width, height=height)
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
if selected_metrics == 'Etat de santé systeme':
    lin1, lin2 = st.beta_columns((6, 6))
    width = 1200
    height = 600
    with lin1:
        
        #selected_feature = st.selectbox(
        #        label="Choisissez un capteur de télémétrie", options=['volt', 'rotate', 'vibration', 'pressure']
        #        )

        #st.write(selected_feature + " pour toutes les machines")
        
        selected_machine1 = st.slider("Choisissez une machine pour le voltage", 1, 100)
        st.write('volt' + " pour la machine " + str(selected_machine1))
        plot = telemetry[
            telemetry.machineID == selected_machine1][["datetime", 'volt']].set_index("datetime")

        fig = px.line(plot, width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        selected_machine2 = st.slider("Choisissez une machine pour la rotation", 1, 100)
        st.write('rotate' + " pour la machine " + str(selected_machine2))
        plot = telemetry[
            telemetry.machineID == selected_machine2][["datetime", 'rotate']].set_index("datetime")

        fig = px.line(plot, width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

    #--------------------------------- ---------------------------------  ---------------------------------
    #--------------------------------- telemetry machine
    #--------------------------------- ---------------------------------  ---------------------------------

    with lin2:

        selected_machine3 = st.slider("Choisissez une machine pour la vibration", 1, 100)
        st.write('vibration' + " pour la machine " + str(selected_machine3))
        plot = telemetry[
            telemetry.machineID == selected_machine3][["datetime", 'vibration']].set_index("datetime")

        fig = px.line(plot, width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        selected_machine4 = st.slider("Choisissez une machine pour la pression", 1, 100)
        st.write('pressure' + " pour la machine " + str(selected_machine4))
        plot = telemetry[
            telemetry.machineID == selected_machine4][["datetime", 'pressure']].set_index("datetime")

        fig = px.line(plot, width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)




    
