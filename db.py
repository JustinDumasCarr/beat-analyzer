"""
SQLite persistence layer for Beat Analyzer jobs.

Provides thread-safe job storage that survives server restarts.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class JobDatabase:
    """Thread-safe SQLite database for job persistence."""

    _local = threading.local()

    def __init__(self, db_path: str = "beat_analyzer.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (created if not exists)
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'queued',
                    filename TEXT,
                    file_path TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    webhook_url TEXT,
                    webhook_fired INTEGER DEFAULT 0,
                    progress_pct INTEGER DEFAULT 0,
                    progress_stage TEXT,
                    fast INTEGER DEFAULT 0,
                    trim INTEGER DEFAULT 0,
                    stems INTEGER DEFAULT 0,
                    waveform INTEGER DEFAULT 1,
                    vocals INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

                -- Enable WAL mode for better concurrent read performance
                PRAGMA journal_mode=WAL;
            """)

    @contextmanager
    def _get_conn(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    def create_job(
        self,
        job_id: str,
        filename: str,
        file_path: str,
        webhook_url: Optional[str] = None,
        fast: bool = False,
        trim: bool = False,
        stems: bool = False,
        waveform: bool = True,
        vocals: bool = True
    ) -> Dict:
        """
        Create a new job record.

        Args:
            job_id: Unique job identifier
            filename: Original filename
            file_path: Path to uploaded file
            webhook_url: Optional URL for completion callback
            fast: Skip ML model
            trim: Create trimmed versions
            stems: Enable stem separation
            waveform: Include waveform data
            vocals: Include vocal detection

        Returns:
            Created job record as dict
        """
        created_at = datetime.now().isoformat()

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, status, filename, file_path, created_at,
                    webhook_url, fast, trim, stems, waveform, vocals,
                    progress_pct, progress_stage
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'queued')
            """, (
                job_id, filename, file_path, created_at,
                webhook_url, int(fast), int(trim), int(stems),
                int(waveform), int(vocals)
            ))

        return {
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "file_path": file_path,
            "created_at": created_at,
            "webhook_url": webhook_url,
            "fast": fast,
            "trim": trim,
            "stems": stems,
            "waveform": waveform,
            "vocals": vocals,
            "progress_pct": 0,
            "progress_stage": "queued"
        }

    def update_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """
        Update job status and optionally set result or error.

        Args:
            job_id: Job identifier
            status: New status (queued, processing, completed, failed)
            result: Analysis result dict (for completed status)
            error: Error message (for failed status)
        """
        completed_at = datetime.now().isoformat() if status in ('completed', 'failed') else None
        result_json = json.dumps(result) if result else None

        with self._get_conn() as conn:
            if status == 'completed':
                conn.execute("""
                    UPDATE jobs SET
                        status = ?,
                        completed_at = ?,
                        result_json = ?,
                        progress_pct = 100,
                        progress_stage = 'complete'
                    WHERE job_id = ?
                """, (status, completed_at, result_json, job_id))
            elif status == 'failed':
                conn.execute("""
                    UPDATE jobs SET
                        status = ?,
                        completed_at = ?,
                        error = ?
                    WHERE job_id = ?
                """, (status, completed_at, error, job_id))
            else:
                conn.execute("""
                    UPDATE jobs SET status = ? WHERE job_id = ?
                """, (status, job_id))

    def update_progress(self, job_id: str, progress_pct: int, stage: str):
        """
        Update job progress.

        Args:
            job_id: Job identifier
            progress_pct: Progress percentage (0-100)
            stage: Current processing stage name
        """
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs SET progress_pct = ?, progress_stage = ?
                WHERE job_id = ?
            """, (progress_pct, stage, job_id))

    def get_job(self, job_id: str) -> Optional[Dict]:
        """
        Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job record as dict, or None if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list_jobs(
        self,
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        List jobs with optional filtering.

        Args:
            limit: Maximum number of jobs to return
            status: Filter by status

        Returns:
            List of job records (without full result)
        """
        with self._get_conn() as conn:
            if status:
                cursor = conn.execute("""
                    SELECT job_id, status, filename, created_at, completed_at,
                           progress_pct, progress_stage, error
                    FROM jobs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (status, limit))
            else:
                cursor = conn.execute("""
                    SELECT job_id, status, filename, created_at, completed_at,
                           progress_pct, progress_stage, error
                    FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE job_id = ?",
                (job_id,)
            )

        return cursor.rowcount > 0

    def mark_webhook_fired(self, job_id: str, success: bool):
        """
        Mark webhook as fired.

        Args:
            job_id: Job identifier
            success: Whether webhook delivery succeeded
        """
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs SET webhook_fired = ?
                WHERE job_id = ?
            """, (1 if success else 2, job_id))

    def get_webhook_url(self, job_id: str) -> Optional[str]:
        """Get webhook URL for a job."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT webhook_url FROM jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()

        return row['webhook_url'] if row else None

    def cleanup_expired(self, days: int = 7) -> int:
        """
        Delete jobs older than specified days.

        Args:
            days: Number of days to keep jobs

        Returns:
            Number of jobs deleted
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute("""
                DELETE FROM jobs WHERE created_at < ?
            """, (cutoff,))

        return cursor.rowcount

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a database row to a job dict."""
        d = dict(row)

        # Parse result JSON if present
        if d.get('result_json'):
            d['result'] = json.loads(d['result_json'])
            del d['result_json']
        else:
            d['result'] = None
            if 'result_json' in d:
                del d['result_json']

        # Convert integer bools back to Python bools
        for key in ('fast', 'trim', 'stems', 'waveform', 'vocals', 'webhook_fired'):
            if key in d:
                d[key] = bool(d[key])

        return d
