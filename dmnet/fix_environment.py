"""
fix_environment.py
==================
Fixes ALL compatibility issues for DMNet on Windows:

  Problem 1: NumPy 2.4.4 → TensorFlow 2.13 needs NumPy < 2.0
  Problem 2: protobuf 4.x → TensorFlow 2.13 needs protobuf 3.20.x

Run from inside your venv:
    python fix_environment.py
"""

import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=False)

print("=" * 60)
print("  DMNet — Environment Compatibility Fixer")
print("=" * 60)

# Step 1: Wipe conflicting packages completely
print("\n[1/4] Removing conflicting packages...")
run(f'"{sys.executable}" -m pip uninstall -y numpy protobuf tensorflow tensorflow-intel tensorflow-cpu')

# Step 2: Install pinned NumPy FIRST (must be before TF)
print("\n[2/4] Installing NumPy 1.x ...")
run(f'"{sys.executable}" -m pip install "numpy==1.26.4"')

# Step 3: Install pinned protobuf
print("\n[3/4] Installing protobuf 3.20.3 ...")
run(f'"{sys.executable}" -m pip install "protobuf==3.20.3"')

# Step 4: Install TensorFlow (will use already-installed numpy 1.26.4)
print("\n[4/4] Installing TensorFlow 2.13.0 ...")
run(f'"{sys.executable}" -m pip install "tensorflow==2.13.0"')

# Verify
print("\n" + "=" * 60)
print("  Verifying installation...")
print("=" * 60)
result = subprocess.run(
    f'"{sys.executable}" -c "import numpy; import tensorflow; print(\'NumPy:\', numpy.__version__); print(\'TensorFlow:\', tensorflow.__version__)"',
    shell=True, capture_output=True, text=True
)
if result.returncode == 0:
    print(result.stdout)
    print("✅ All good! Now run:  python train.py")
else:
    print("❌ Still an issue:")
    print(result.stderr[-800:])
    print("\nTry manually:")
    print('  pip install "numpy==1.26.4" "protobuf==3.20.3" "tensorflow==2.13.0"')
