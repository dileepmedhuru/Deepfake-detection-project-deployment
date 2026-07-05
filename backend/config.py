import os
from datetime import timedelta
from pathlib import Path
import secrets

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # ── Security ──────────────────────────────────────────────
    SECRET_KEY     = os.environ.get('SECRET_KEY')     or secrets.token_hex(32)
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or secrets.token_hex(32)
    _using_generated_keys = not os.environ.get('SECRET_KEY')

    # ── Database ───────────────────────────────────────────────
    DATABASE_PATH = BASE_DIR / 'database' / 'deepfake.db'
    db_url = os.environ.get('DATABASE_URL')
    if db_url and db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    SQLALCHEMY_DATABASE_URI = db_url or f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── JWT ────────────────────────────────────────────────────
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)

    # ── File Upload ────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
    ALLOWED_IMAGE_MIMES = {'image/jpeg','image/png','image/gif','image/bmp','image/webp'}
    ALLOWED_VIDEO_MIMES = {'video/mp4','video/avi','video/quicktime',
                           'video/x-matroska','video/x-msvideo','video/x-ms-wmv'}

    # ── File Cleanup (APScheduler) ─────────────────────────────
    UPLOAD_RETENTION_DAYS = int(os.environ.get('UPLOAD_RETENTION_DAYS', 7))

    # ── ML Model ───────────────────────────────────────────────
    MODEL_PATH = BASE_DIR / 'ml_models' / 'cnn_model.h5'

    # ── Email (Flask-Mail) ─────────────────────────────────────
    MAIL_SERVER   = os.environ.get('MAIL_SERVER',   'smtp.gmail.com')
    MAIL_PORT     = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS  = os.environ.get('MAIL_USE_TLS',  'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@deepfake.com')
    MAIL_ENABLED  = bool(os.environ.get('MAIL_USERNAME', ''))

    # ── Password Reset Token ───────────────────────────────────
    PASSWORD_RESET_EXPIRY_MINUTES = 30

    @staticmethod
    def init_app(app):
        db_dir = BASE_DIR / 'database'
        db_dir.mkdir(exist_ok=True)
        if Config._using_generated_keys:
            print('⚠️  SECRET_KEY / JWT_SECRET_KEY not set — sessions reset on restart.')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    @staticmethod
    def init_app(app):
        Config.init_app(app)
        if not os.environ.get('SECRET_KEY') or not os.environ.get('JWT_SECRET_KEY'):
            raise RuntimeError('SECRET_KEY and JWT_SECRET_KEY must be set in production.')


config = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'default':     DevelopmentConfig,
}
