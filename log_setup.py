import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent / "laventra.log"

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"


class ColouredFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG:    DIM    + "[DBG] " + RESET,
        logging.INFO:     CYAN   + "[INF] " + RESET,
        logging.WARNING:  YELLOW + "[WRN] " + RESET,
        logging.ERROR:    RED    + "[ERR] " + RESET,
        logging.CRITICAL: BOLD   + RED + "[CRT] " + RESET,
    }

    def format(self, record):
        prefix   = self.COLOURS.get(record.levelno, "")
        time_str = self.formatTime(record, "%H:%M:%S")
        msg      = record.getMessage()
        if "✅" in msg:
            msg = GREEN + msg + RESET
        elif "❌" in msg:
            msg = RED + msg + RESET
        elif "⚠️" in msg:
            msg = YELLOW + msg + RESET
        elif "🚗" in msg:
            msg = WHITE + BOLD + msg + RESET
        return f"{DIM}{time_str}{RESET}  {prefix}{msg}"


class PlainFormatter(logging.Formatter):
    def format(self, record):
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level    = record.levelname[:3]
        return f"{time_str} [{level}] {record.getMessage()}"


def setup(debug: bool = False) -> logging.Logger:
    root  = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColouredFormatter())
    root.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(PlainFormatter())
    root.addHandler(fh)

    return logging.getLogger("laventra")


def get() -> logging.Logger:
    return logging.getLogger("laventra")