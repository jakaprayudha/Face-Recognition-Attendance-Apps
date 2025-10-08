import os
import mysql.connector
from dotenv import load_dotenv

# Muat variabel dari file .env
load_dotenv()

def get_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            autocommit=True
        )
        return conn
    except mysql.connector.Error as err:
        print(f"[DB ERROR] {err}")
        return None