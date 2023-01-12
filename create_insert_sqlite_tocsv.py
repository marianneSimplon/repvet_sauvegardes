import pandas as pd
import sqlite3

# establish database connection


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)

    return conn

# create sqlite database tables


def create_prediction(conn, predictions):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO predictions(predictions_review_text, predictions_recommended, predictions_review_score)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, predictions)
    conn.commit()
    return cur.lastrowid

# save data in csv file repvet.csv


def save_db_to_csv(conn):
    repvet = pd.read_sql_query('SELECT * from predictions', conn)
    repvet.to_csv('./prediction_extractions/repvet.csv', index=False)
