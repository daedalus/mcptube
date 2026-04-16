#!/usr/bin/env python3
"""Save YouTube watch history video IDs to a file."""

import json

ids = json.load(open("/tmp/history.json"))
print("\n".join(ids))
