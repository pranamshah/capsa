The SQLite database (vibration.db) is created automatically in backend/ when
mqtt_subscriber.py runs. Export a dataset here for your report or offline retraining
with:  sqlite3 vibration.db ".dump" > dump.sql
