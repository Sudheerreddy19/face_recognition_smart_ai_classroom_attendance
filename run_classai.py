
import os
import sys
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

log_file = open(
    BASE_DIR / "classai.log",
    "a",
    encoding="utf-8",
    buffering=1
)

# Provide valid output streams for pythonw.exe
sys.stdout = log_file
sys.stderr = log_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(log_file)],
    force=True
)

try:
    import uvicorn

    logging.info("Starting ClassAI Face Service...")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        log_config=None,
        access_log=True
    )

except Exception:
    logging.exception("ClassAI startup failed")

finally:
    logging.info("ClassAI service stopped.")
    log_file.flush()
    log_file.close()
