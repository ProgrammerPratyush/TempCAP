import pandas as pd
import sqlite3

def load_excel_to_db(excel_file_path):
    # Read the Excel file
    df = pd.read_excel(excel_file_path)

    # Connect to SQLite database
    conn = sqlite3.connect('soiltype.db')

    # Load the DataFrame into the SQLite table
    df.to_sql('soiltype', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    print("Data loaded into the SQLite database.")

if __name__ == "__main__":
    excel_file_path = 'sltp.xlsx'  # Replace with the path to your Excel file
    load_excel_to_db(excel_file_path)
