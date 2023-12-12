from datetime import datetime, timedelta
from pathlib import Path

def create_exp_dir(logs_dir: Path, exp_name: str):
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d_%H-%M")
    exp_dir = logs_dir / exp_name / date_string
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir