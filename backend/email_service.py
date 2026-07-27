"""
Email Service – Flask-Mail wrapper
Sends: welcome, detection result, password reset emails.
All sends are silent-fail so the app keeps working without SMTP configured.
"""

from flask_mail import Mail, Message
from flask import current_app, request as flask_request
import jwt, os
from datetime import datetime, timedelta

mail = Mail()


def _enabled() -> bool:
    return current_app.config.get('MAIL_ENABLED', False)


def _base_url() -> str:
    """Return the public base URL of the app (no trailing slash)."""
    # 1. Explicit env var — set this in HF Spaces / Render / Railway secrets
    env_url = os.environ.get('APP_URL', '').rstrip('/')
    if env_url:
        return env_url
    # 2. Derive from current request host if inside request context
    try:
        return flask_request.host_url.rstrip('/')
    except RuntimeError:
        return 'http://localhost:5000'


# ── Token helpers ──────────────────────────────────────────────────────────

def generate_reset_token(user_id: int) -> str:
    expiry = datetime.utcnow() + timedelta(
        minutes=current_app.config.get('PASSWORD_RESET_EXPIRY_MINUTES', 30)
    )
    payload = {'user_id': user_id, 'purpose': 'password_reset', 'exp': expiry}
    return jwt.encode(payload, current_app.config['JWT_SECRET_KEY'], algorithm='HS256')


def verify_reset_token(token: str):
    """Return user_id if token is valid, else None."""
    try:
        payload = jwt.decode(
            token, current_app.config['JWT_SECRET_KEY'], algorithms=['HS256']
        )
        if payload.get('purpose') != 'password_reset':
            return None
        return payload.get('user_id')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ── Email senders ──────────────────────────────────────────────────────────

def send_welcome_email(user_email: str, full_name: str):
    if not _enabled():
        print('[Email] Skipped — MAIL_ENABLED is False (MAIL_USERNAME not set in environment).')
        return
    try:
        msg = Message(
            subject='Welcome to Deepfake Detector!',
            recipients=[user_email],
        )
        msg.html = f"""
        <div style="font-family:sans-serif;max-width:560px;margin:auto;">
          <div style="background:linear-gradient(135deg,#06b6d4,#0891b2);
                      padding:2rem;border-radius:12px 12px 0 0;text-align:center;">
            <h1 style="color:white;margin:0;">🔍 Deepfake Detector</h1>
          </div>
          <div style="background:#f8fafc;padding:2rem;border-radius:0 0 12px 12px;
                      border:1px solid #e2e8f0;">
            <h2 style="color:#1e293b;">Welcome, {full_name}!</h2>
            <p style="color:#475569;">Your account has been created successfully.
               You can now upload images and videos to detect deepfakes.</p>
            <a href="{_base_url()}/dashboard.html"
               style="display:inline-block;background:#06b6d4;color:white;
                      padding:.75rem 1.5rem;border-radius:8px;text-decoration:none;
                      margin-top:1rem;">
              Go to Dashboard
            </a>
            <p style="color:#94a3b8;font-size:.85rem;margin-top:2rem;">
              If you didn't create this account, please ignore this email.
            </p>
          </div>
        </div>"""
        mail.send(msg)
    except Exception as e:
        print(f'[Email] Welcome email failed: {e}')


def send_detection_result_email(user_email: str, full_name: str,
                                 file_name: str, result: str,
                                 confidence: float, detection_id: int):
    if not _enabled():
        print('[Email] Skipped — MAIL_ENABLED is False (MAIL_USERNAME not set in environment).')
        return
    try:
        is_fake   = result.lower() == 'fake'
        color     = '#dc2626' if is_fake else '#059669'
        icon      = '⚠️' if is_fake else '✅'
        label     = 'FAKE DETECTED' if is_fake else 'AUTHENTIC'

        msg = Message(
            subject=f'Detection Result: {label} – {file_name}',
            recipients=[user_email],
        )
        msg.html = f"""
        <div style="font-family:sans-serif;max-width:560px;margin:auto;">
          <div style="background:linear-gradient(135deg,#06b6d4,#0891b2);
                      padding:2rem;border-radius:12px 12px 0 0;text-align:center;">
            <h1 style="color:white;margin:0;">🔍 Detection Result</h1>
          </div>
          <div style="background:#f8fafc;padding:2rem;border-radius:0 0 12px 12px;
                      border:1px solid #e2e8f0;">
            <p style="color:#475569;">Hi {full_name}, your analysis is complete.</p>
            <div style="background:{color};color:white;border-radius:12px;
                        padding:1.5rem;text-align:center;margin:1.5rem 0;">
              <div style="font-size:2.5rem;">{icon}</div>
              <h2 style="margin:.5rem 0;">{label}</h2>
              <p style="opacity:.9;margin:0;">{file_name}</p>
            </div>
            <table style="width:100%;border-collapse:collapse;">
              <tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:.75rem;color:#64748b;">Confidence</td>
                <td style="padding:.75rem;font-weight:700;color:#1e293b;">{confidence}%</td>
              </tr>
              <tr>
                <td style="padding:.75rem;color:#64748b;">Result</td>
                <td style="padding:.75rem;font-weight:700;color:{color};">{result.upper()}</td>
              </tr>
            </table>
            <a href="{_base_url()}/results.html?id={detection_id}"
               style="display:inline-block;background:#06b6d4;color:white;
                      padding:.75rem 1.5rem;border-radius:8px;text-decoration:none;
                      margin-top:1.5rem;">
              View Full Details
            </a>
          </div>
        </div>"""
        mail.send(msg)
    except Exception as e:
        print(f'[Email] Result email failed: {e}')


def send_password_reset_email(user_email: str, full_name: str, reset_token: str):
    if not _enabled():
        print('[Email] Skipped — MAIL_ENABLED is False (MAIL_USERNAME not set in environment).')
        return
    try:
        reset_url = f'{_base_url()}/reset-password.html?token={reset_token}'
        msg = Message(
            subject='Reset Your Password – Deepfake Detector',
            recipients=[user_email],
        )
        msg.html = f"""
        <div style="font-family:sans-serif;max-width:560px;margin:auto;">
          <div style="background:linear-gradient(135deg,#06b6d4,#0891b2);
                      padding:2rem;border-radius:12px 12px 0 0;text-align:center;">
            <h1 style="color:white;margin:0;">🔒 Password Reset</h1>
          </div>
          <div style="background:#f8fafc;padding:2rem;border-radius:0 0 12px 12px;
                      border:1px solid #e2e8f0;">
            <h2 style="color:#1e293b;">Hi {full_name},</h2>
            <p style="color:#475569;">
              We received a request to reset your password.
              Click the button below within <strong>30 minutes</strong>.
            </p>
            <a href="{reset_url}"
               style="display:inline-block;background:#06b6d4;color:white;
                      padding:.75rem 1.5rem;border-radius:8px;text-decoration:none;
                      margin-top:1rem;">
              Reset Password
            </a>
            <p style="color:#94a3b8;font-size:.85rem;margin-top:2rem;">
              If you didn't request this, you can safely ignore this email.
              Your password will not change.
            </p>
          </div>
        </div>"""
        mail.send(msg)
    except Exception as e:
        print(f'[Email] Reset email failed: {e}')
