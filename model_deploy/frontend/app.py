import streamlit as st
from multiapp import MultiApp
from apps import home, evaluations_models

app = MultiApp()

# Add all your application here
app.add_app("Predição", home.app)
app.add_app("Performance", evaluations_models.app)

# The main app
app.run()