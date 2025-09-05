"""
Application Settings and Configuration
"""
import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Database Configuration
    database_url: str = Field(default="sqlite:///./compliance.db", env="DATABASE_URL")
    
    # Supabase Configuration
    supabase_url: Optional[str] = Field(default=None, env="SUPABASE_URL")
    supabase_key: Optional[str] = Field(default=None, env="SUPABASE_KEY")
    supabase_jwt_secret: Optional[str] = Field(default=None, env="SUPABASE_JWT_SECRET")
    supabase_db_url: Optional[str] = Field(default=None, env="SUPABASE_DB_URL")
        
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Security
    secret_key: str = Field(default="dev-secret-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    llm_max_workers: int = Field(default=5, env="LLM_MAX_WORKERS")
    llm_timeout: int = Field(default=60, env="LLM_TIMEOUT")
    
    # Vector Database Configuration
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    vector_collection_name: str = Field(default="policy_chunks", env="VECTOR_COLLECTION_NAME")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="./logs/compliance.log", env="LOG_FILE")
    
    # Email Configuration
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    email_from: Optional[str] = Field(default=None, env="EMAIL_FROM")
    
    # External Services
    webhook_url: Optional[str] = Field(default=None, env="WEBHOOK_URL")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    
    # File Upload Configuration
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(default=["pdf", "txt", "doc", "docx", "xlsx"])
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Compliance Configuration
    default_materiality_bps: int = Field(default=100, env="DEFAULT_MATERIALITY_BPS")
    breach_notification_delay_minutes: int = Field(default=5, env="BREACH_NOTIFICATION_DELAY_MINUTES")
    auto_resolve_breaches: bool = Field(default=False, env="AUTO_RESOLVE_BREACHES")
    
    # Development/Testing
    testing: bool = Field(default=False, env="TESTING")
    mock_llm: bool = Field(default=False, env="MOCK_LLM")
    skip_vector_store: bool = Field(default=False, env="SKIP_VECTOR_STORE")
    
    model_config = {
        "env_file": [".env.local", ".env"],
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }
    
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.debug or self.api_reload
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.is_development
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def has_llm_config(self) -> bool:
        """Check if LLM configuration is available"""
        return bool(self.openai_api_key or self.anthropic_api_key)
    
    @property
    def has_email_config(self) -> bool:
        """Check if email configuration is available"""
        return all([
            self.smtp_host, 
            self.smtp_username, 
            self.smtp_password, 
            self.email_from
        ])
    
    def create_upload_dir(self):
        """Create upload directory if it doesn't exist"""
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def create_log_dir(self):
        """Create log directory if it doesn't exist"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()