import os
import re
from pathlib import Path

try:
    from google.colab import drive
except ImportError:
    drive = None


DIR_NAME = "llm_assignment_3_submission-{student_id}"


def is_valid_student_id(input_string: str) -> bool:
    """
    Checks if the input string matches the pattern 'XX-XXX-XXX' where X is any digit.

    Args:
        input_string (str): The string to check

    Returns:
        bool: True if the string matches the pattern, False otherwise
    """
    pattern = r"^\d{2}-\d{3}-\d{3}$"

    # Use re.match to check if the entire string matches the pattern
    return bool(re.match(pattern, input_string))


def get_solution_path(student_id: str, save_on_drive: bool) -> Path:
    if save_on_drive and drive is None:
        raise ValueError(
            "You are not running this code in Google Colab. If you want to save your solution on your Google Drive, please run this code in Google Colab."
        )
    if save_on_drive and drive is not None:
        if not os.path.exists("/content/drive"):
            drive.mount("/content/drive")
        path = Path(f"/content/drive/MyDrive/{DIR_NAME.format(student_id=student_id)}")
    else:
        path = Path(f"./{DIR_NAME.format(student_id=student_id)}")
    path.mkdir(parents=True, exist_ok=True)
    return path
