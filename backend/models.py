from database import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model):
    """User model"""
    __tablename__ = 'users'
    __table_args__ = {'extend_existing': True}

    id         = db.Column(db.Integer, primary_key=True)
    full_name  = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(100), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)   # hashed
    is_admin   = db.Column(db.Boolean, default=False)
    is_verified= db.Column(db.Boolean, default=True)
    force_password_change = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow)

    detections = db.relationship(
        'Detection', backref='user', lazy=True, cascade='all, delete-orphan'
    )

    # ---------- password helpers ----------
    def set_password(self, raw_password: str):
        """Hash and store password."""
        self.password = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password, raw_password)

    # ---------- serialisation ----------
    def to_dict(self):
        return {
            'id':           self.id,
            'full_name':    self.full_name,
            'email':        self.email,
            'is_admin':     self.is_admin,
            'is_verified':  self.is_verified,
            'force_password_change': self.force_password_change,
            'created_at':   self.created_at.isoformat() if self.created_at else None,
        }


class Detection(db.Model):
    """Detection history model"""
    __tablename__ = 'detection_history'
    __table_args__ = {'extend_existing': True}

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    file_name       = db.Column(db.String(255), nullable=False)
    file_type       = db.Column(db.String(10),  nullable=False)   # 'image' | 'video'
    file_path       = db.Column(db.String(500), nullable=False)
    result          = db.Column(db.String(30),  nullable=False)   # 'real' | 'fake' | 'ai_generated'
    confidence      = db.Column(db.Float,       nullable=False)
    processing_time = db.Column(db.Float,       nullable=False)
    is_demo         = db.Column(db.Boolean, default=False)        # NEW: demo-mode flag
    extra_data      = db.Column('metadata', db.Text)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':              self.id,
            'user_id':         self.user_id,
            'file_name':       self.file_name,
            'file_type':       self.file_type,
            # file_path intentionally omitted – internal server path
            'result':          self.result,
            'confidence':      round(self.confidence, 2),
            'processing_time': round(self.processing_time, 2),
            'is_demo':         self.is_demo,
            'metadata':        self.extra_data,
            'created_at':      self.created_at.isoformat() if self.created_at else None,
        }
