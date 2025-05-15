#!/usr/bin/env python
"""ActionGraph ML model and data suite."""

# package version
from actiongraph.version import __version__

# silence the pyflakes syntax checker
assert __version__ or True

# Import and register actiongraph
from actiongraph import ActionGraph

__all__ = ["actiongraph"]

# End of file
