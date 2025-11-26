#!/usr/bin/env python
import subprocess
import os

py = r"D:\Sem_3\Project\GIS\lulc_env\Scripts\python.exe"
print(f"Testing interpreter: {py}")
print(f"Exists: {os.path.exists(py)}")

# Test 1: Direct import
print("\nTest 1: Direct import")
result = os.system(f'{py} -c "import torch, rasterio; print(\'OK\')"')
print(f"Direct exec result: {result}")

# Test 2: Subprocess Popen (as plugin does)
print("\nTest 2: Subprocess Popen (as plugin)")
cmd = [py, '-c', 'import torch, rasterio; print("OK")']
print(f"Command: {cmd}")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = proc.communicate(timeout=30)
print(f"Return code: {proc.returncode}")
print(f"Stdout: {out.decode('utf-8', errors='ignore')}")
if err:
    print(f"Stderr: {err.decode('utf-8', errors='ignore')}")
if proc.returncode == 0:
    print("✓ SUCCESS - Packages detected")
else:
    print("✗ FAILED - Packages not detected")
