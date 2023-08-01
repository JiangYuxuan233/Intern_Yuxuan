from django.apps import AppConfig
from django.conf import settings
import os
import pickle


class YuxuanDjiangoConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "yuxuan_djiango"
    path = os.path.join(settings.MODELS,"Un-Copy1.py")
    with open(path,"rb") as pickled:
        data = pickle.load(pickled)
    model = data["model"]
    vectorizer = data["vectorizer"]
        
