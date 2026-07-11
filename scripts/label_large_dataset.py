# DEPRECATED: Use scripts/score_pseudo_labels.py instead.
#
# This legacy script referenced PATHS keys that no longer exist (raw_data,
# finetuned_model, output_labeled). score_pseudo_labels.py scores an existing
# pseudo-labeled CSV with the 4K seed model and adds confidence columns.

import sys
import warnings

warnings.warn(
    "label_large_dataset.py is deprecated. Use scripts/score_pseudo_labels.py instead.",
    DeprecationWarning,
    stacklevel=1,
)

print("This script is deprecated.")
print("Use: python scripts/score_pseudo_labels.py")
print("See README.md -> Confidence Threshold Experiments")
sys.exit(1)
