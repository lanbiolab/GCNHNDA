# encoding: utf-8
# @email: enoche.chow@gmail.com
"""
############################

Note: Top-K evaluation metrics have been removed.
Binary classification evaluation is now used instead.
See utils/binary_evaluator.py for the new evaluation metrics.
"""

from logging import getLogger

import numpy as np


# Top-K metrics have been deprecated in favor of binary classification metrics
# Please use BinaryEvaluator in utils/binary_evaluator.py

# Legacy metrics_dict kept for compatibility (empty)
metrics_dict = {}