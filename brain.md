# Brain Knowledge Base & Architecture Specification (brain.md)

This document contains the complete technical specification, architectural layout, configuration parameters, and execution design of the **Vision Attendance System (`d:\vt_application_v6.2_cpu`)**.

---

## 🏗️ 1. Core Architecture & Component Hierarchy

### A. Dual-Layered Guaranteed Pending Snapshot Processing Pipeline ([core/snapshot_manager.py](file:///d:/vt_application_v6.2_cpu/core/snapshot_manager.py) & [core/multiprocess_handler.py](file:///d:/vt_application_v6.2_cpu/core/multiprocess_handler.py))
- **Layer 1: Central Snapshot Pipeline Scanner (`scan_pending_snapshots`)**:
  - `scan_pending_snapshots()` in [core/snapshot_manager.py:L311](file:///d:/vt_application_v6.2_cpu/core/snapshot_manager.py#L311) continuously scans `data/pending_snapshots/` every 2 seconds, sorting files by modification time (`os.path.getmtime`) and pushing any un-processed snapshot $> 0.5\text{s}$ old directly into AI worker task queues without restrictive set blocking.
- **Layer 2: Worker-Level Idle & Recovery Scavenging (`scavenge_orphaned_files`)**:
  - In [core/multiprocess_handler.py:L275](file:///d:/vt_application_v6.2_cpu/core/multiprocess_handler.py#L275), `AttendanceWorker` processes directly scavenge `data/pending_snapshots/` during startup and 30-second idle sweeps, immediately processing any remaining pending image JPEGs.
  - Guarantees 100% of captured pending snapshots are processed and deleted from `data/pending_snapshots/` under all operating conditions.

---

## 📊 2. Performance & Detection Efficiency Metrics

| Component / Subsystem | Behavior | Performance Impact |
| :--- | :--- | :--- |
| **Pending Snapshot Delivery** | Dual-layered (Pipeline Scanner + Idle Worker Scavenger) | 100% processing guarantee |
| **Queue Dispatch** | Non-blocking `put_nowait` sorted by timestamp | Zero queue blocking or orphaned JPEGs |
