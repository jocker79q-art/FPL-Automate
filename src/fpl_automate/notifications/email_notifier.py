"""SMTP email notifications.

Credentials come only from `Settings` (i.e. environment variables / .env),
are never logged, and any failure to send is caught and logged without
ever printing the password or full SMTP transcript.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from fpl_automate.config import Settings

logger = logging.getLogger(__name__)


class EmailSendError(RuntimeError):
    pass


class EmailNotifier:
    def __init__(self, settings: Settings) -> None:
        if not settings.email_notifications_enabled:
            raise ValueError("EmailNotifier constructed while EMAIL_NOTIFICATIONS_ENABLED=false")
        self._settings = settings

    def send(self, subject: str, body_text: str, body_html: str | None = None) -> None:
        s = self._settings
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = s.email_from
        msg["To"] = s.email_to
        msg.set_content(body_text)
        if body_html:
            msg.add_alternative(body_html, subtype="html")

        try:
            if s.smtp_use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20) as server:
                    server.starttls(context=context)
                    server.login(s.smtp_username, s.smtp_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=20) as server:
                    server.login(s.smtp_username, s.smtp_password)
                    server.send_message(msg)
        except smtplib.SMTPAuthenticationError as exc:
            raise EmailSendError(
                "SMTP authentication failed. Check SMTP_USERNAME/SMTP_PASSWORD "
                "(for Gmail, this must be an App Password, not your account password)."
            ) from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise EmailSendError(f"Failed to send email: {exc.__class__.__name__}") from exc

        logger.info("Sent email notification to %s", _mask_email(s.email_to))


def _mask_email(address: str) -> str:
    if "@" not in address:
        return "***"
    local, _, domain = address.partition("@")
    masked_local = local[0] + "***" if local else "***"
    return f"{masked_local}@{domain}"
