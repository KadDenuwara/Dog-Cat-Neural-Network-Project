from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
Base = declarative_base()
engine = create_engine("sqlite:///predictions.db")  # Use SQLite database (or replace with your database URI)
Session = sessionmaker(bind=engine)
session = Session()

# Define table structure
class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String, nullable=False)  # Image file name
    file_path = Column(String, nullable=False)  # Path to the saved image
    prediction = Column(String, nullable=False)  # Prediction result
    timestamp = Column(DateTime, default=datetime.utcnow)  # Upload timestamp

# Create the table
Base.metadata.create_all(engine)
