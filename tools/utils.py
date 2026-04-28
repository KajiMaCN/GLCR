import os


def get_abspath(path):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    return os.path.abspath(os.path.join(root_dir, path))
