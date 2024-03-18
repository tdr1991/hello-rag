import psycopg2

db_name = "vector_db"
host = "localhost"
passwd = "postgres"
port = "5432"
user = "postgres"
conn = psycopg2.connect(dbname="postgres",
                        host=host,
                        user=user,
                        password=passwd)

conn.autocommit = True
with conn.cursor() as cur:
    cur.execute("drop database if exists {}".format(db_name))
    cur.execute("create database {}".format(db_name))


conn.close()