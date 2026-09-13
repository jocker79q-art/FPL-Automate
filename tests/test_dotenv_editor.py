from __future__ import annotations

from pathlib import Path

from fpl_automate.dotenv_editor import read_env_values, update_env_file


def test_read_missing_file_returns_empty(tmp_path: Path):
    assert read_env_values(tmp_path / "does-not-exist.env") == {}


def test_read_ignores_comments_and_blank_lines(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("# a comment\n\nFPL_TEAM_ID=123\n# EMAIL_TO=old@example.com\n", encoding="utf-8")
    assert read_env_values(path) == {"FPL_TEAM_ID": "123"}


def test_update_creates_file_when_missing(tmp_path: Path):
    path = tmp_path / ".env"
    update_env_file(path, {"FPL_TEAM_ID": "9242093"})
    assert read_env_values(path) == {"FPL_TEAM_ID": "9242093"}


def test_update_replaces_existing_key_in_place(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("# header comment\nFPL_TEAM_ID=111\nMAX_TRANSFER_RISK=4\n", encoding="utf-8")
    update_env_file(path, {"FPL_TEAM_ID": "9242093"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# header comment"
    assert "FPL_TEAM_ID=9242093" in lines
    assert "MAX_TRANSFER_RISK=4" in lines
    assert lines.index("FPL_TEAM_ID=9242093") < lines.index("MAX_TRANSFER_RISK=4")


def test_update_appends_new_key(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("FPL_TEAM_ID=9242093\n", encoding="utf-8")
    update_env_file(path, {"EMAIL_TO": "me@example.com"})
    assert read_env_values(path) == {"FPL_TEAM_ID": "9242093", "EMAIL_TO": "me@example.com"}


def test_update_preserves_unrelated_keys_and_comments(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nFPL_TEAM_ID=111\nSMTP_HOST=smtp.gmail.com\n\n# another comment\n",
        encoding="utf-8",
    )
    update_env_file(path, {"FPL_TEAM_ID": "222"})
    text = path.read_text(encoding="utf-8")
    assert "# comment" in text
    assert "# another comment" in text
    assert "SMTP_HOST=smtp.gmail.com" in text
    assert "FPL_TEAM_ID=222" in text


def test_update_multiple_keys_at_once(tmp_path: Path):
    path = tmp_path / ".env"
    update_env_file(
        path,
        {
            "FPL_TEAM_ID": "9242093",
            "EMAIL_NOTIFICATIONS_ENABLED": "true",
            "SMTP_USERNAME": "me@example.com",
        },
    )
    values = read_env_values(path)
    assert values["FPL_TEAM_ID"] == "9242093"
    assert values["EMAIL_NOTIFICATIONS_ENABLED"] == "true"
    assert values["SMTP_USERNAME"] == "me@example.com"
