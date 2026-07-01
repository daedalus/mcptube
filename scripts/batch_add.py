#!/usr/bin/env python3
"""Batch process YouTube watch history with mcptube."""

import os
import json
import subprocess
import sys
import random

proxy = "http://172.29.99.99:3128"
model = "openrouter/openrouter/auto"
randomize = False

def main():
    ids = json.load(open("scripts/history.json"))
    print(f"Processing {len(ids)} videos...", file=sys.stderr)

    if randomize: random.shuffle(ids)

    for i, vid in enumerate(ids):
        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"[{i + 1}/{len(ids)}] {vid}", file=sys.stderr, end=" ... ", flush=True)

        try:
            #os.system(f"mcptube --proxy {proxy} --cookies-from-browser chromium --show-frame-stats --verbose --debug add --reprocess {url}")
            os.system(f"mcptube --proxy {proxy} --cookies-from-browser chromium --show-frame-stats --verbose --model {model}  add --reprocess {url}")

            """
            result = subprocess.run(
                [
                    "mcptube",
                    "--proxy",
                    "http://172.29.99.99:3128",
                    "--cookies-from-browser",
                    "chromium",
                    "--show-frame-stats",
                    "--verbose",
                    "--debug",
                    "add",
                    "--reprocess",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            # Show last non-empty line

            lines = [l for l in result.stderr.strip().split("\n") if l.strip()]
            #last = lines[-1] if lines else "done"
            if  len(lines) > 0:
                print("ERR:",lines)


            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            #last = lines[-1] if lines else "done"
            if len(lines) > 0:
                print("OUT:",lines)
            """


        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
