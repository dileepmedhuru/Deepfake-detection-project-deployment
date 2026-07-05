import os
from app import create_app

# Render and production deployment WSGI entry point
config_name = os.environ.get('FLASK_CONFIG') or 'production'
app = create_app(config_name)
