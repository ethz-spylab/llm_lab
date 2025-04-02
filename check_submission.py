# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///
import os
from pathlib import Path
import sys
import numpy as np
import re
from zipfile import ZipFile, Path as ZipPath


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


Q1_GENS_FILENAME = "Q1_gens.txt"
Q1_GUESSES_FILENAME = "Q1_guesses.npy"


class InvalidSubmissionError(Exception):
    pass


def has_specific_subdirectory(parent_path: ZipPath, subdir_name: str) -> bool:
    subdir_path = parent_path / subdir_name
    return subdir_path.exists() and subdir_path.is_dir()


def check_q1(path: ZipPath) -> None:
    # check generations file
    gens_file = path / Q1_GENS_FILENAME
    if not gens_file.exists():
        raise InvalidSubmissionError(f"{Q1_GENS_FILENAME} not found in the submission")
    with gens_file.open() as f:
        gens = f.readlines()
        if len(gens) != 60:
            raise InvalidSubmissionError(
                f"{Q1_GENS_FILENAME} should contain exactly 60 elements"
            )

    # check guesses file
    guesses_file = path / Q1_GUESSES_FILENAME
    if not guesses_file.exists():
        raise InvalidSubmissionError(
            f"{Q1_GUESSES_FILENAME} not found in the submission"
        )
    with guesses_file.open("rb") as f:
        guesses = np.load(f)
        print(guesses)
        if not np.issubdtype(guesses.dtype, np.integer):
            raise InvalidSubmissionError(
                f"{Q1_GUESSES_FILENAME} should contain integers only."
            )
        if not np.all(guesses > 0) or not np.all(guesses < 5):
            raise InvalidSubmissionError(
                f"{Q1_GUESSES_FILENAME} should contain integers between 1 and 4 (both inclusive) only."
            )
        if guesses.shape != (80,):
            raise InvalidSubmissionError(
                f"{Q1_GUESSES_FILENAME} should contain exactly 80 elements."
            )


def check_q2(path: ZipPath) -> None:
    # Implement your check for question 2 here
    pass


def check_q3(path: ZipPath) -> None:
    # Implement your check for question 3 here
    pass


def check_submission(zip_file_path: str, student_id: str) -> None:
    if not os.path.exists(zip_file_path):
        raise InvalidSubmissionError(f"Submission file {zip_file_path} not found.")
    with ZipFile(zip_file_path, "r") as zip_file:
        root_path = ZipPath(zip_file)
        if has_specific_subdirectory(root_path, student_id):
            root_path = root_path / student_id
        check_q1(root_path)
        check_q2(root_path)
        check_q3(root_path)
    print("Submission is valid.")


def main(student_id: str) -> None:
    zip_file_path = f"{student_id}.zip"
    check_submission(zip_file_path, student_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python check_submission.py student_id\n - student_id: the student ID on your legi (looks like 'nn-nnn-nnn')"
        )
        sys.exit(1)
    student_id = sys.argv[1]
    if not is_valid_student_id(student_id):
        print("Student ID must be in the 'nn-nnn-nnn' format.")
        sys.exit(1)
    main(student_id)
