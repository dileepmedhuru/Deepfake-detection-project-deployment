"""
File Cleanup Scheduler
Deletes uploaded files older than UPLOAD_RETENTION_DAYS (default 7).
Runs automatically when the Flask app starts.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import os
import time
from pathlib import Path

_scheduler = None


def _cleanup_old_files(upload_folder: str, retention_days: int):
    """Delete files older than retention_days from upload subfolders."""
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    freed_bytes = 0

    for subfolder in ('images', 'videos'):
        folder = os.path.join(upload_folder, subfolder)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    size = os.path.getsize(fpath)
                    os.remove(fpath)
                    deleted += 1
                    freed_bytes += size
            except Exception as e:
                print(f'[Cleanup] Could not delete {fpath}: {e}')

    if deleted:
        freed_mb = freed_bytes / (1024 * 1024)
        print(f'[Cleanup] Deleted {deleted} files, freed {freed_mb:.1f} MB '
              f'(older than {retention_days} days)')
    else:
        print(f'[Cleanup] No files older than {retention_days} days found.')


def start_scheduler(app):
    """Start the background cleanup scheduler. Call once from create_app()."""
    global _scheduler

    upload_folder   = app.config.get('UPLOAD_FOLDER', 'uploads')
    retention_days  = app.config.get('UPLOAD_RETENTION_DAYS', 7)

    _scheduler = BackgroundScheduler(daemon=True)

    # Run every day at 02:00 AM
    _scheduler.add_job(
        func=lambda: _cleanup_old_files(upload_folder, retention_days),
        trigger=CronTrigger(hour=2, minute=0),
        id='file_cleanup',
        name='Delete old uploaded files',
        replace_existing=True,
    )

    _scheduler.start()
    print(f'[OK] File cleanup scheduler started '
          f'(retention: {retention_days} days, runs daily at 02:00).')

    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
