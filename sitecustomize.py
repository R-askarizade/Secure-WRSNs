from __future__ import annotations

import ssl
import certifi


_original_create_default_context = ssl.create_default_context


def _create_default_context_with_certifi(
    purpose=ssl.Purpose.SERVER_AUTH,
    *,
    cafile=None,
    capath=None,
    cadata=None,
):
    if cafile is None and capath is None and cadata is None:
        cafile = certifi.where()

    return _original_create_default_context(
        purpose=purpose,
        cafile=cafile,
        capath=capath,
        cadata=cadata,
    )


ssl.create_default_context = _create_default_context_with_certifi
ssl._create_default_https_context = _create_default_context_with_certifi
