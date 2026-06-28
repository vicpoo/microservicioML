from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "microservicioMLL"
    database_url: str = "sqlite:///./app.db"

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
