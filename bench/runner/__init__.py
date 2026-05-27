"""Recall trace-harvester runtime."""

import litellm

# Per-model param compatibility: gpt-5 family rejects temperature != 1.0,
# o1/o3 reject system messages, etc. Tell litellm to silently drop
# unsupported params per model rather than raise. The harvester takes
# comparative measurements across models; uniform request shape across
# the matrix matters more than enforcing every kwarg.
litellm.drop_params = True
