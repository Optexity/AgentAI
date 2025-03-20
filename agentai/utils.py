import logging
import os
import sys

from computergym.actions.action import string_to_action_type
from pydantic import BaseModel

from .prompts import Response


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = "agent.log",
    log_to_console: bool = True,
    log_path: str = None,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create log directory if it doesn't exist
    if log_path:
        os.makedirs(log_path, exist_ok=True)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
    )

    # File handler (logs errors to file)
    if log_file and log_path:
        file_handler = logging.FileHandler(os.path.join(log_path, log_file))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)  # Ensure it captures all logs
        logger.addHandler(file_handler)

    # Console handler (optional)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)  # Ensure it captures all logs
        logger.addHandler(console_handler)

    # Ensure uncaught exceptions go to all handlers
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.exception(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = exception_handler

    try:
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    except Exception as e:
        pass

    return logger


def action_to_response(action: BaseModel):
    return Response(
        action_name=action.__class__.__name__, action_params=action.model_dump()
    )


def response_to_action(response: Response) -> BaseModel:
    action_params = response.action_params
    action_object = string_to_action_type[response.action_name]
    action = action_object.model_validate(action_params)
    return action
