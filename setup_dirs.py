#!/usr/bin/env python3
import os

dirs = [
    '/Users/josedab/Code/code-gen/genesis/website/docs/getting-started',
    '/Users/josedab/Code/code-gen/genesis/website/docs/concepts',
    '/Users/josedab/Code/code-gen/genesis/website/docs/guides',
    '/Users/josedab/Code/code-gen/genesis/website/docs/api',
    '/Users/josedab/Code/code-gen/genesis/website/docs/advanced',
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}")

print("Done!")
