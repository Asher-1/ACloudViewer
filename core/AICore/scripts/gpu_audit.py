#!/usr/bin/env python3
"""Single-process NVIDIA measurement audit shared by validation probes.

This module neither schedules scenarios nor downloads assets. Native process
latencies remain the timing source. Driver memory is a sampled high-water
mark, not an allocator's exact peak. Unsupported or incomplete telemetry can
never produce a controlled-measurement PASS.
"""

import json
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET


def _number(text):
    try:
        return float(text.split()[0])
    except (AttributeError, IndexError, ValueError):
        return None


def parse_devices(xml):
    root = ET.fromstring(xml)
    devices = []
    for node in root.findall("gpu"):
        processes = []
        for process in node.findall("processes/process_info"):
            processes.append({"pid": int(process.findtext("pid")),
                              "type": process.findtext("type"),
                              "used_mib": _number(process.findtext("used_memory"))})
        devices.append({"uuid": node.findtext("uuid"),
                        "name": node.findtext("product_name"),
                        "driver": root.findtext("driver_version"),
                        "utilization_pct": _number(node.findtext("utilization/gpu_util")),
                        "used_mib": _number(node.findtext("fb_memory_usage/used")),
                        "sm_clock_mhz": _number(node.findtext("clocks/sm_clock")),
                        "memory_clock_mhz": _number(node.findtext("clocks/mem_clock")),
                        "temperature_c": _number(node.findtext("temperature/gpu_temp")),
                        "power_w": _number(node.findtext("power_readings/power_draw")),
                        "processes": processes})
    if not devices or any(not d["uuid"] or d["utilization_pct"] is None or
                          d["used_mib"] is None for d in devices):
        raise ValueError("missing GPU identity, utilization or memory telemetry")
    return devices


def parse_activity(output):
    if not output.strip():
        raise ValueError("empty GPU process activity telemetry")
    rows = []
    for line in output.splitlines():
        columns = line.split()
        if not columns or columns[0].startswith("#"):
            continue
        if len(columns) < 4:
            raise ValueError("malformed GPU process activity")
        if columns[1] == "-":  # driver's empty-device sentinel
            continue
        if columns[2] not in ("C", "G", "C+G"):
            raise ValueError("unknown GPU process type")
        rows.append({"pid": int(columns[1]), "type": columns[2],
                     "sm_pct": _number(columns[3])})
    return rows


def external_activity(sample, owner_pid=None, graphics_threshold=5):
    busy = set()
    for device in sample["devices"]:
        busy.update(p["pid"] for p in device["processes"]
                    if "C" in p["type"] and p["pid"] != owner_pid)
    busy.update(p["pid"] for p in sample["activity"] if p["pid"] != owner_pid
                and p["sm_pct"] is not None and p["sm_pct"] > graphics_threshold)
    return sorted(busy)


def sample_driver():
    def query(args):
        return subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, check=True, timeout=5).stdout
    return {"devices": parse_devices(query(["-q", "-x"])),
            "activity": parse_activity(query(["pmon", "-c", "1", "-s", "u"]))}


def summarize_samples(samples, owner_pid, errors=()):
    active = sorted({pid for s in samples for pid in external_activity(s, owner_pid)})
    owned = [p["used_mib"] for s in samples for d in s["devices"]
             for p in d["processes"] if p["pid"] == owner_pid and p["used_mib"] is not None]
    return {"passed": bool(samples and owned and not active and not errors),
            "external_active_pids": active, "errors": list(errors),
            "sample_count": len(samples), "owner_memory_sample_count": len(owned),
            "sampled_process_peak_mib": max(owned) if owned else None,
            "sampled_device_peak_mib": max((d["used_mib"] for s in samples
                                             for d in s["devices"]), default=None)}


def run_audited(command, audit_path, *, strict=False, idle_timeout=60, **kwargs):
    """Run one native probe and retain telemetry, including rejected samples.

    Compute contexts from other processes are always rejected. Desktop
    graphics must stay at or below 5% SM in pmon. A strict run additionally
    requires three pre-run samples at <=10% total utilization. No process is
    killed, ignored through an allow-list, or assigned to the measurement.
    """
    if idle_timeout <= 0:
        raise ValueError("GPU idle timeout must be positive")
    audit = {"schema": 1, "audit_id": uuid.uuid4().hex, "strict": strict,
             "graphics_sm_threshold_pct": 5, "idle_total_threshold_pct": 10,
             "memory_scope": "sampled driver process/device memory; not exact allocator peak",
             "waiting_samples": [], "samples": []}
    errors = []
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        deadline = time.monotonic() + idle_timeout
        consecutive = 0
        while True:
            try:
                sample = sample_driver()
            except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
                errors.append(str(error))
                if strict:
                    raise RuntimeError("GPU audit telemetry unavailable") from error
                break
            audit["waiting_samples"].append(sample)
            idle = (not external_activity(sample) and
                    all(d["utilization_pct"] <= 10 for d in sample["devices"]))
            consecutive = consecutive + 1 if idle else 0
            if not strict or consecutive >= 3:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU did not become idle before measurement")
            time.sleep(min(.2, max(0, deadline - time.monotonic())))

        begin = time.monotonic()
        process = subprocess.Popen(command, **kwargs)
        audit["owner_pid"] = process.pid
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    sample = sample_driver()
                    sample["seconds"] = time.monotonic() - begin
                    audit["samples"].append(sample)
                except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
                    errors.append(str(error))
                stop.wait(.05)

        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise
        finally:
            stop.set()
            worker.join()
        audit.update(summarize_samples(audit["samples"], process.pid, errors))
        audit["idle_passed"] = consecutive >= 3
        audit["passed"] = bool(audit["passed"] and audit["idle_passed"])
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        return completed, audit
    except BaseException as error:
        audit.update(passed=False, error=str(error), errors=errors)
        raise
    finally:
        # A scenario's public report path is reused by process-level repeats.
        # Keep each raw audit independently, including failed idle waits.
        archive = audit_path.with_suffix(".d") / (audit["audit_id"] + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        audit["archive_path"] = str(archive)
        document = json.dumps(audit, indent=2) + "\n"
        with archive.open("x") as stream:
            stream.write(document)
        audit_path.write_text(document)
