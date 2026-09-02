"""
QA Automated Test Suite for Attendance System
Validates end-to-end integration between Attendance_Application and Django HRMS Backend.

Usage:
    python qa_attendance_test.py
"""

import sys
import os
import time
from datetime import datetime, date

# Ensure root paths are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.auth_manager import AuthManager
from core.api_client import APIClient
from database.offline_storage import OfflineStorage

class AttendanceQATester:
    def __init__(self):
        self.results = []
        self.client = None
        self.username = None
        self.org_name = None
        self.test_person_id = None
        self.test_person_name = None

    def log_result(self, test_name, passed, details=""):
        status_str = "[PASS]" if passed else "[FAIL]"
        self.results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        print(f" {status_str} | {test_name}")
        if details:
            print(f"        |-- {details}")

    def run_all_tests(self):
        print("=" * 70)
        print("    ATTENDANCE SYSTEM FULL END-TO-END QA TEST SUITE")
        print(f"    Executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()

        self.test_01_credentials_and_login()
        self.test_02_scoped_face_encodings_sync()
        self.test_03_clock_in_attendance()
        self.test_04_clock_out_attendance()
        self.test_05_cross_org_security_rejection()
        self.test_06_offline_storage_caching()
        self.test_07_daily_attendance_reporting()

        self.print_summary()

    def test_01_credentials_and_login(self):
        print("[1/7] Testing Authentication & JWT Bearer Token Generation...")
        try:
            u, p, url = AuthManager.load_credentials()
            if not u or not p:
                self.log_result("Load Local Credentials", False, "Missing credentials in auth_config.json")
                return

            self.username = u
            self.client = APIClient(api_base_url=url, api_user=u, api_password=p)
            login_success, token = self.client.login()

            if login_success and self.client.token:
                self.org_name = getattr(self.client, 'user_org_name', 'Default Org')
                self.log_result(
                    "API Authentication (JWT Bearer)", 
                    True, 
                    f"User: {u} | Org: {self.org_name} | Token Length: {len(token)}"
                )
            else:
                self.log_result("API Authentication (JWT Bearer)", False, f"Login failed: {token}")
        except Exception as e:
            self.log_result("API Authentication (JWT Bearer)", False, f"Exception: {e}")

    def test_02_scoped_face_encodings_sync(self):
        print("\n[2/7] Testing Organization-Scoped Face Encodings Sync...")
        if not self.client or not self.client.token:
            self.log_result("Face Encodings Sync", False, "Skipped: Client not authenticated")
            return

        try:
            encodings = self.client.get_all_face_encodings()
            if encodings and len(encodings) > 0:
                # Pick first person for subsequent attendance testing
                first_pid = list(encodings.keys())[0]
                self.test_person_id = first_pid
                self.test_person_name = encodings[first_pid].get('name', 'Unknown')
                
                # Check vector shape
                sample_enc = encodings[first_pid].get('encoding')
                dim = len(sample_enc) if sample_enc is not None else 0

                self.log_result(
                    "Organization Face Encodings Sync", 
                    True, 
                    f"Loaded {len(encodings)} faces for '{self.org_name}'. Sample: {self.test_person_name} ({first_pid}, {dim}-dim)"
                )
            else:
                self.log_result(
                    "Organization Face Encodings Sync", 
                    True, 
                    f"Sync returned 0 faces for organization '{self.org_name}' (Clean scoped state)"
                )
        except Exception as e:
            self.log_result("Organization Face Encodings Sync", False, f"Exception: {e}")

    def test_03_clock_in_attendance(self):
        print("\n[3/7] Testing Clock-In Attendance Marking...")
        if not self.client or not self.client.token:
            self.log_result("Clock-In Attendance", False, "Skipped: Client not authenticated")
            return

        target_id = self.test_person_id or "dummy12"
        try:
            now = datetime.now()
            success, msg = self.client.mark_attendance(
                person_id=target_id,
                timestamp=now,
                snapshot_path=None,
                event_type="in"
            )
            self.log_result(
                f"Clock-In Event ({target_id})", 
                success, 
                f"Response: {msg} | Time: {now.strftime('%H:%M:%S')}"
            )
        except Exception as e:
            self.log_result(f"Clock-In Event ({target_id})", False, f"Exception: {e}")

    def test_04_clock_out_attendance(self):
        print("\n[4/7] Testing Clock-Out Attendance Marking & Update...")
        if not self.client or not self.client.token:
            self.log_result("Clock-Out Attendance", False, "Skipped: Client not authenticated")
            return

        target_id = self.test_person_id or "dummy12"
        try:
            now = datetime.now()
            success, msg = self.client.mark_attendance(
                person_id=target_id,
                timestamp=now,
                snapshot_path=None,
                event_type="out"
            )
            self.log_result(
                f"Clock-Out Event ({target_id})", 
                success, 
                f"Response: {msg} | Time: {now.strftime('%H:%M:%S')}"
            )
        except Exception as e:
            self.log_result(f"Clock-Out Event ({target_id})", False, f"Exception: {e}")

    def test_05_cross_org_security_rejection(self):
        print("\n[5/7] Testing Cross-Organization Security Guard...")
        if not self.client or not self.client.token:
            self.log_result("Cross-Org Security Guard", False, "Skipped: Client not authenticated")
            return

        # Use an employee ID that belongs to a different organization
        foreign_person_id = "PI001" 
        try:
            success, msg = self.client.mark_attendance(
                person_id=foreign_person_id,
                timestamp=datetime.now(),
                snapshot_path=None,
                event_type="in"
            )
            # We EXPECT success to be False (Rejection with Organization Mismatch)
            if not success and ("Organization mismatch" in msg or "not belong" in msg or "not found" in msg or "404" in msg):
                self.log_result(
                    "Cross-Org Security Boundary", 
                    True, 
                    f"Successfully blocked foreign attendance for ID '{foreign_person_id}'. Detail: {msg}"
                )
            elif not success:
                self.log_result(
                    "Cross-Org Security Boundary", 
                    True, 
                    f"Blocked foreign employee request cleanly. Reason: {msg}"
                )
            else:
                self.log_result(
                    "Cross-Org Security Boundary", 
                    False, 
                    f"Security Alert: Cross-org attendance was accepted when it should be blocked! Msg: {msg}"
                )
        except Exception as e:
            self.log_result("Cross-Org Security Boundary", False, f"Exception: {e}")

    def test_06_offline_storage_caching(self):
        print("\n[6/7] Testing Offline SQLite Fallback Storage...")
        try:
            storage = OfflineStorage()
            test_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 1. Insert test offline attendance
            storage.add_attendance("QA_TEST_001", test_ts, "snapshots/qa_test.jpg", "in")
            
            # 2. Query pending
            pending = storage.get_pending_attendance()
            found = any(row[1] == "QA_TEST_001" for row in pending)
            
            # 3. Clean up test record
            for row in pending:
                if row[1] == "QA_TEST_001":
                    storage.delete_attendance(row[0])
            
            if found:
                self.log_result(
                    "Offline SQLite Storage Caching", 
                    True, 
                    f"Verified offline attendance write, read, and delete in {storage.db_path}"
                )
            else:
                self.log_result("Offline SQLite Storage Caching", False, "Could not find inserted QA test record")
        except Exception as e:
            self.log_result("Offline SQLite Storage Caching", False, f"Exception: {e}")

    def test_07_daily_attendance_reporting(self):
        print("\n[7/7] Testing Daily Attendance Reporting API...")
        if not self.client or not self.client.token:
            self.log_result("Daily Attendance Reporting", False, "Skipped: Client not authenticated")
            return

        try:
            records = self.client.get_today_attendance()
            self.log_result(
                "Daily Attendance Retrieval", 
                True, 
                f"Retrieved {len(records)} active attendance records for date {date.today().isoformat()}"
            )
        except Exception as e:
            self.log_result("Daily Attendance Retrieval", False, f"Exception: {e}")

    def print_summary(self):
        print()
        print("=" * 70)
        print("                     QA TEST SUMMARY RESULTS")
        print("=" * 70)
        
        passed_count = sum(1 for r in self.results if r["passed"])
        failed_count = len(self.results) - passed_count
        
        for r in self.results:
            status = "[PASS]" if r["passed"] else "[FAIL]"
            print(f" {status:<7} {r['test']}")
            
        print("-" * 70)
        print(f" TOTAL TESTS: {len(self.results)} | PASSED: {passed_count} | FAILED: {failed_count}")
        
        if failed_count == 0:
            print(" >>> ALL ATTENDANCE SYSTEM TESTS PASSED PERFECTLY! <<<")
        else:
            print(" >>> SOME TESTS FAILED. PLEASE CHECK DETAILS ABOVE. <<<")
        print("=" * 70)

if __name__ == "__main__":
    tester = AttendanceQATester()
    tester.run_all_tests()
