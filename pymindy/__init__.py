"""pymindy package root.

STUB (added to make the package importable): the original package exposed a
module-level ``logger`` that pymindy.training.utils imports as
``from .. import logger``.  That file was not carried into this copy of the
repo, so a plain stdlib logger is provided here.  It only needs ``.info()``.
"""

import logging
import sys

logger = logging.getLogger("pymindy")

# Only configure once; a notebook re-import must not stack duplicate handlers.
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

__all__ = ["logger"]
