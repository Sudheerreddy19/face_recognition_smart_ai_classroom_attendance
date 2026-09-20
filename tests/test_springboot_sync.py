"""
tests/test_springboot_sync.py
Quick test to verify Spring Boot connection and attendance sync.

Usage:
    python tests/test_springboot_sync.py

Steps tested:
  1. Health check  - is Spring Boot reachable?
  2. Student check - does Spring Boot know our student?
  3. Attendance    - can we POST a test attendance record?
"""

import os
import sys
from datetime import date, time as dtime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import get_settings
settings = get_settings()

BASE_URL    = settings.spring_boot_base_url
ATT_EP      = settings.spring_boot_attendance_endpoint
STU_EP      = settings.spring_boot_student_endpoint
TIMEOUT     = settings.spring_boot_timeout

print()
print("=" * 60)
print("  SPRING BOOT SYNC TEST")
print("=" * 60)
print(f"  Base URL    : {BASE_URL}")
print(f"  Attendance  : {ATT_EP}")
print(f"  Student     : {STU_EP}")
print(f"  Timeout     : {TIMEOUT}s")
print()

import requests

# ── 1. Health check ───────────────────────────────────────────
print("[TEST 1] Health check ...")
try:
    r = requests.get(f"{BASE_URL}/actuator/health", timeout=5)
    if r.ok:
        print(f"  PASS  Spring Boot is UP  ({r.status_code})")
        try:
            print(f"  Status: {r.json().get('status', 'unknown')}")
        except Exception:
            pass
    else:
        print(f"  WARN  HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"  FAIL  Cannot reach Spring Boot: {e}")
    print()
    print("  Make sure Spring Boot is running:")
    print("    cd your-spring-boot-project")
    print("    mvn spring-boot:run   OR   java -jar target/your-app.jar")
    print()
    print("  If running on a different port, update .env:")
    print("    SPRING_BOOT_BASE_URL=http://localhost:YOUR_PORT")
    sys.exit(1)

# ── 2. Student verification ───────────────────────────────────
print()
print("[TEST 2] Student verification ...")
TEST_STUDENT = "23FE1A0424"
try:
    r = requests.get(f"{BASE_URL}{STU_EP}/{TEST_STUDENT}", timeout=TIMEOUT)
    if r.ok:
        print(f"  PASS  Student {TEST_STUDENT} found in Spring Boot")
        try:
            data = r.json()
            print(f"  Name: {data.get('name', data.get('studentName', '?'))}")
        except Exception:
            pass
    elif r.status_code == 404:
        print(f"  WARN  Student {TEST_STUDENT} not found in Spring Boot DB")
        print("        (Attendance POST may still work depending on your SB logic)")
    else:
        print(f"  WARN  HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"  WARN  Student check error: {e}")

# ── 3. Attendance POST ────────────────────────────────────────
print()
print("[TEST 3] POST attendance record ...")
payload = {
    "registerNumber":       TEST_STUDENT,
    "studentName":          "Test Student",
    "date":                 date.today().isoformat(),
    "time":                 "10:30:00",
    "subject":              "TEST_SUBJECT",
    "teacherId":            "TEACHER01",
    "session":              "CSE-A",
    "recognitionDistance":  0.350,
    "status":               "PRESENT",
    "source":               "PYTHON_FACE_SERVICE",
}

print(f"  Payload: {payload}")
print()

try:
    r = requests.post(
        f"{BASE_URL}{ATT_EP}",
        json=payload,
        timeout=TIMEOUT,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    if r.ok:
        print(f"  PASS  Attendance accepted! HTTP {r.status_code}")
        try:
            print(f"  Response: {r.json()}")
        except Exception:
            print(f"  Response: {r.text[:300]}")
    else:
        print(f"  FAIL  HTTP {r.status_code}")
        print(f"  Body : {r.text[:500]}")
        print()
        print("  If 404 -> check SPRING_BOOT_ATTENDANCE_ENDPOINT in .env")
        print("  If 400 -> check what fields your Spring Boot expects")
        print("  If 405 -> endpoint exists but method not allowed")
except Exception as e:
    print(f"  FAIL  {e}")

print()
print("=" * 60)
print("  DONE")
print("=" * 60)
print()
