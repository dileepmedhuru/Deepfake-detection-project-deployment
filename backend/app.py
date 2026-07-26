from dotenv import load_dotenv
load_dotenv()
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import config
from database import db

import os

limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day","60 per hour"])

def create_app(config_name='development'):
    app = Flask(__name__, static_folder='../frontend', static_url_path='')
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    db.init_app(app)
    CORS(app)
    limiter.init_app(app)

    # Email
    from email_service import mail
    mail.init_app(app)

    # Upload dirs
    os.makedirs(os.path.join(app.config.get('UPLOAD_FOLDER','uploads'),'images'), exist_ok=True)
    os.makedirs(os.path.join(app.config.get('UPLOAD_FOLDER','uploads'),'videos'), exist_ok=True)

    # Blueprints
    from auth_routes      import auth_bp
    from detection_routes import detection_bp
    from admin_routes     import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(admin_bp)

    # Apply rate limits to sensitive endpoints
    limiter.limit("5 per minute")(auth_bp)

    # DB + seed
    with app.app_context():
        try:
            db.create_all()
            _seed_default_admin()
            print('[OK] Database ready.')
        except Exception as e:
            print(f'[ERROR] Database error: {e}')

    # File cleanup scheduler
    try:
        from cleanup import start_scheduler
        start_scheduler(app)
    except Exception as e:
        print(f'[WARN] Cleanup scheduler not started: {e}')

    # Frontend
    @app.route('/')
    def index():
        return send_from_directory('../frontend', 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        full = os.path.join('../frontend', path)
        if os.path.exists(full):
            return send_from_directory('../frontend', path)
        return send_from_directory('../frontend', 'index.html')

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy'}), 200

    # Error handlers
    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Resource not found.'}), 404
        return send_from_directory('../frontend', 'index.html'), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({'error': 'Method not allowed.'}), 405

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({'error': 'File too large. Maximum size is 100 MB.'}), 413

    @app.errorhandler(429)
    def rate_limited(_e):
        return jsonify({'error': 'Too many requests. Please slow down.'}), 429

    @app.errorhandler(500)
    def internal_error(_e):
        return jsonify({'error': 'Internal server error.'}), 500

    return app


def _seed_default_admin():
    from models import User
    admin_email    = os.environ.get('ADMIN_EMAIL', 'admin@deepfake.com')
    admin_password = os.environ.get('ADMIN_PASSWORD', None)
    if User.query.filter_by(email=admin_email).first():
        return
    if admin_password is None:
        import secrets, string
        alphabet = string.ascii_letters + string.digits + '!@#$%'
        admin_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        print('='*60)
        print('Default admin account created:')
        print(f'   Email:    {admin_email}')
        print(f'   Password: {admin_password}')
        print('   Change this at profile.html after first login!')
        print('='*60)
    admin = User(full_name='Admin User', email=admin_email,
                 is_admin=True, is_verified=True, force_password_change=True)
    admin.set_password(admin_password)
    from database import db
    db.session.add(admin)
    db.session.commit()


if __name__ == '__main__':
    app = create_app('development')
    print('[OK] Starting on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=True)
