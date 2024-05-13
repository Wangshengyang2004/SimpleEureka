import sys


def check_system_encoding():
    """Check the system encoding and return False if not UTF-8."""
    if sys.getdefaultencoding() != 'utf-8':
        return False