"""
fix_protobuf_error.py
=====================
Run this ONCE to fix:
  "cannot import name 'runtime_version' from 'google.protobuf'"

This error happens because protobuf 4.x+ broke compatibility
with TensorFlow 2.x on Windows.

Usage:
    python fix_protobuf_error.py
"""

import subprocess
import sys


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


print("=" * 60)
print("  DMNet — Fixing protobuf / TensorFlow conflict")
print("=" * 60)

# Step 1: Uninstall conflicting packages
run(f"{sys.executable} -m pip uninstall -y protobuf tensorflow tensorflow-intel")

# Step 2: Install pinned protobuf FIRST
run(f"{sys.executable} -m pip install protobuf==3.20.3")

# Step 3: Install compatible TensorFlow
run(f"{sys.executable} -m pip install tensorflow==2.13.0")

# Step 4: Reinstall rest of requirements
run(f"{sys.executable} -m pip install -r requirements.txt")

print("\n" + "=" * 60)
print("  ✅ Fix applied!")
print("  Now run:  python train.py")
print("=" * 60)
