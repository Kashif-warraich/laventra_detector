"""
Logging setup — rotating file handler + colourised stdout (only when a TTY).
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

import config

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, YELLOW, GREEN, CYAN, WHITE = (
    "\033[91m", "\033[93m", "\033[92m", "\033[96m", "\033[97m",
)


class _ColouredFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG:    DIM    + "[DBG] " + RESET,
        logging.INFO:     CYAN   + "[INF] " + RESET,
        logging.WARNING:  YELLOW + "[WRN] " + RESET,
        logging.ERROR:    RED    + "[ERR] " + RESET,
        logging.CRITICAL: BOLD + RED + "[CRT] " + RESET,
    }

    def format(self, record):
        prefix = self.COLOURS.get(record.levelno, "")
        time_str = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        if "✅" in msg:
            msg = GREEN + msg + RESET
        elif "❌" in msg:
            msg = RED + msg + RESET
        elif "⚠️" in msg:
            msg = YELLOW + msg + RESET
        elif "🚗" in msg:
            msg = WHITE + BOLD + msg + RESET
        return f"{DIM}{time_str}{RESET}  {prefix}{msg}"


class _PlainFormatter(logging.Formatter):
    def format(self, record):
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname[:3]
        return f"{time_str} [{level}] {record.getMessage()}"


def setup(*, debug: bool = False) -> logging.Logger:
    root = logging.getLogger()
    # Avoid re-attaching handlers if setup() is called twice
    for h in list(root.handlers):
        root.removeHandler(h)

    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    # Only emit ANSI escapes when stdout is an actual terminal — otherwise
    # nohup/systemd capture logs full of escape sequences (audit L6).
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    ch.setFormatter(_ColouredFormatter() if is_tty else _PlainFormatter())
    root.addHandler(ch)

    fh = RotatingFileHandler(
        config.LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    root.addHandler(fh)

    return logging.getLogger("laventra")


def get() -> logging.Logger:
    return logging.getLogger("laventra")
