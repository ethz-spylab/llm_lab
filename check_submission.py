# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///
import base64
import os
import sys
import numpy as np
import re
from pathlib import Path
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


# Q1 parameters
Q1_GENS_FILENAME = "Q1_gens.txt"
Q1_GUESSES_FILENAME = "Q1_guesses.npy"
Q1_GENS_LEN = 60
Q1_GUESSES_SHAPE = (80,)
Q1_GUESSES_RANGE = (1, 4)

# Q2 parameters
Q2_API_KEY_FILENAME = "Q2_key.txt"
Q2_CODE_FILENAME = "Q2_code"
Q2_CODE_EXTENSIONS = [".py", ".ipynb"]
Q2_PVALUE_FILENAME = "Q2_pvalue.npy"
Q2_BOOL_FILENAME = "Q2_bool.npy"
Q2_ARRAYS_SHAPE = (140,)
Q2_KEY_LEN = 50

# Q3 parameters
Q3_GUESSES_FILENAME = "Q3_guesses.txt"
Q3_GUESSES_LEN = 21


class InvalidSubmissionError(Exception):
    pass


def has_specific_subdirectory(parent_path: ZipPath, subdir_name: str) -> bool:
    subdir_path = parent_path / subdir_name
    return subdir_path.exists() and subdir_path.is_dir()


def is_valid_token_urlsafe(token: str, key_len: int) -> bool:
    try:
        # Add padding if needed
        padded = token + "=" * (4 - len(token) % 4) if len(token) % 4 else token
        # Replace URL-safe chars with standard base64 chars
        padded = padded.replace("-", "+").replace("_", "/")
        # Decode and check length
        decoded = base64.b64decode(padded)
        return len(decoded) == key_len
    except Exception:
        return False


def check_q1(path: ZipPath) -> None:
    # check generations file
    gens_file = path / Q1_GENS_FILENAME
    if not gens_file.exists():
        raise InvalidSubmissionError(f"{Q1_GENS_FILENAME} not found in the submission")
    with gens_file.open() as f:
        gens = f.readlines()
        if len(gens) != Q1_GENS_LEN:
            raise InvalidSubmissionError(
                f"{Q1_GENS_FILENAME} should contain exactly {Q1_GENS_LEN} elements, got {len(gens)}."
            )

    # check guesses file
    guesses_file = path / Q1_GUESSES_FILENAME
    if not guesses_file.exists():
        raise InvalidSubmissionError(
            f"{Q1_GUESSES_FILENAME} not found in the submission"
        )
    with guesses_file.open("rb") as f:
        guesses = np.load(f)
        if not np.issubdtype(guesses.dtype, np.integer):
            raise InvalidSubmissionError(
                f"{Q1_GUESSES_FILENAME} should contain integers only, got {guesses.dtype}."
            )
        if not (
            np.all(guesses >= Q1_GUESSES_RANGE[0])
            and np.all(guesses <= Q1_GUESSES_RANGE[1])
        ):
            raise InvalidSubmissionError(
                f"{Q1_GUESSES_FILENAME} should contain integers between {Q1_GUESSES_RANGE[0]} and {Q1_GUESSES_RANGE[1]} (both inclusive) only."
            )
        if guesses.shape != Q1_GUESSES_SHAPE:
            raise InvalidSubmissionError(
                f"{Q1_GUESSES_FILENAME} should have shape {Q1_GUESSES_SHAPE}, got {guesses.shape}."
            )
    print("Q1 checks passed.")


def check_q2(path: ZipPath) -> None:
    api_key_file = path / Q2_API_KEY_FILENAME
    if not api_key_file.exists():
        raise InvalidSubmissionError(
            f"{Q2_API_KEY_FILENAME} not found in the submission"
        )
    with api_key_file.open() as f:
        api_key = f.read().strip()
        if not is_valid_token_urlsafe(api_key, Q2_KEY_LEN):
            raise InvalidSubmissionError(
                f"{Q2_API_KEY_FILENAME} should contain a base64-encoded string of {Q2_KEY_LEN} bytes."
            )

    # Check the code file
    code_file_exists = False
    for ext in Q2_CODE_EXTENSIONS:
        code_file = path / f"{Q2_CODE_FILENAME}{ext}"
        if code_file.exists():
            if code_file_exists:
                raise InvalidSubmissionError(
                    f"Multiple {Q2_CODE_FILENAME} files found, only one is allowed."
                )
            code_file_exists = True

    if not code_file_exists:
        raise InvalidSubmissionError(
            f"No {Q2_CODE_FILENAME} file found. Please submit a file with one of these extensions: {', '.join(Q2_CODE_EXTENSIONS)}"
        )

    # control p-values file
    for file in [Q2_PVALUE_FILENAME, Q2_BOOL_FILENAME]:
        array_file = path / file
        if not array_file.exists():
            raise InvalidSubmissionError(f"No {file} file found. Please submit it.")
        with array_file.open("rb") as f:
            array = np.load(f)
            if not np.issubdtype(array.dtype, np.integer):
                raise InvalidSubmissionError(
                    f"{file} should contain integers only, got {array.dtype}."
                )
            if array.shape != Q2_ARRAYS_SHAPE:
                raise InvalidSubmissionError(
                    f"{file} should contain an array of shape {Q2_ARRAYS_SHAPE}, got {array.shape}."
                )

    print("Q2 checks passed.")


def check_q3(path: ZipPath) -> None:
    # check generations file
    guesses_file = path / Q3_GUESSES_FILENAME
    if not guesses_file.exists():
        raise InvalidSubmissionError(
            f"{Q3_GUESSES_FILENAME} not found in the submission."
        )
    with guesses_file.open() as f:
        guesses = f.readlines()
        if len(guesses) != Q3_GUESSES_LEN:
            raise InvalidSubmissionError(
                f"{Q3_GUESSES_FILENAME} should contain exactly {Q3_GUESSES_LEN} elements, got {len(guesses)}."
            )


def check_submission(zip_file_path: Path, student_id: str) -> None:
    if not os.path.exists(zip_file_path):
        raise InvalidSubmissionError(f"Submission file {zip_file_path} not found.")
    with ZipFile(zip_file_path, "r") as zip_file:
        root_path = ZipPath(zip_file)
        if has_specific_subdirectory(root_path, zip_file_path.stem):
            root_path = root_path / zip_file_path.stem
        check_q1(root_path)
        check_q2(root_path)
        check_q3(root_path)
    print("Submission is valid.")


def main(student_id: str) -> None:
    zip_file_path = Path(f"llm_assignment_3_submission-{student_id}.zip")
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
