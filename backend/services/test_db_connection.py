import os
import pg8000
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load environment variables
RDS_HOST = os.getenv('RDS_HOST')
RDS_PORT = int(os.getenv('RDS_PORT', 5432))
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')
# RDS_DB = os.getenv('RDS_DB')

def test_db_connection():
    try:
        connection = pg8000.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            # database=RDS_DB
        )
        connection.close()
        print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {str(e)}")

if __name__ == "__main__":
    test_db_connection()