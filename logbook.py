# ---------------------------------------------------
# Version: 20.02.2025
# Author: M. Weber
# ---------------------------------------------------
# ---------------------------------------------------

"""
Dieses Modul enthält Funktionen zum Hinzufügen von Einträgen in ein Logbuch.
"""

from datetime import datetime
import os
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Define constants ----------------------------------
load_dotenv()

mongoClient = MongoClient(os.environ.get('MONGO_URI_PRIVAT'))
database = mongoClient.logbook
coll_log = database.log

# Functions -----------------------------------------
def add_entry(app: str = "default", user: str = "default", text: str = "empty") -> bool:
    """
    Fügt einen neuen Eintrag in das Logbuch ein.

    Args:
        app (str): Der Name der Anwendung, die den Eintrag erstellt. Standard ist "default".
        user (str): Der Benutzername, der den Eintrag erstellt. Standard ist "default".

    Returns:
        bool: True, wenn der Eintrag erfolgreich hinzugefügt wurde, False bei einem Duplikatfehler.
    """
    try:
        coll_log.insert_one({
            'app': app,
            'user': user,
            'text': text,
            'created': datetime.now()
        })
        return True
    except DuplicateKeyError:
        return False