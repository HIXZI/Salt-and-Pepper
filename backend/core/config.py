from pydantic_settings import BaseSettings


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
db_path = os.path.join(BASE_DIR, "audio_forensics.db")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Audio Forensics API"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = f"sqlite:///{db_path}"

    # File Storage
    UPLOAD_DIR: str = "data/samples/"

    # Server
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # Security
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 24 hours

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
