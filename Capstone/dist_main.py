import sqlite3

def create_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('soiltype.db')
    cursor = conn.cursor()

    # Create the table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS soiltype (
        `State Name` TEXT,
        `District` TEXT,
        `Soil Type` TEXT
    )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print("Database and table created.")

if __name__ == "__main__":
    create_database()
