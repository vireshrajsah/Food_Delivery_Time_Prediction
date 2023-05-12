import os
import sys
from src.logging import logging

def get_error_message(error_message, error_details:sys):
    """
    : Params: the error object and error_details from the sys modules traceback object (sys.exc_info).
    : Returns: a DETAILED ERROR MESSAGE with filename and line number.
    """
    _,_,tb_obj = error_details.exc_info()
    file_name = tb_obj.tb_frame.f_code.co_filename
    line_num  = tb_obj.tb_lineno

    custom_error_message = f"Exception logged from file [{file_name}] line no. [{line_num}] with message [{error_message}]"

    return custom_error_message

class CustomException(Exception):
    def __init__(self, error, error_details:sys) -> None:
        super().__init__(error)                               # --
        self.error_message = get_error_message(error, error_details)


    def __str__(self) -> str:
        return self.error_message