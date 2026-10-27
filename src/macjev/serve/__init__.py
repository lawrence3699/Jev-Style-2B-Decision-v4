"""MacJev local server: systemone-compatible decision API, Playground and demos.

Run ``python -m macjev.serve [--release DIR | --fake]``; see docs/demos/API.md.
"""
from .schema import ApiError  # noqa: F401


def create_app(*args, **kwargs):
    from .app import create_app as _create
    return _create(*args, **kwargs)


def build_app(*args, **kwargs):
    from .app import build_app as _build
    return _build(*args, **kwargs)
