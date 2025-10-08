import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",     # ganti sesuai host
        user="root",          # user database
        password="",          # password database
        database="face_absensi"
    )