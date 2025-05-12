#!/usr/bin/env python
##############################################################################
#
# (c) 2025 The Trustees of Columbia University in the City of New York.
# All rights reserved.
#
# File coded by: Sam Andrello.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/8bitsam/actiongraph/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""ActionGraph ML model and data suite."""

# package version
from actiongraph.version import __version__

# silence the pyflakes syntax checker
assert __version__ or True

# Import and register actiongraph
from actiongraph import ActionGraph

__all__ = ["actiongraph"]

# End of file
