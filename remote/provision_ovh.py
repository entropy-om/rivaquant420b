"""Provision an OVHcloud compute-optimized instance for the "quantal"
stage — a deliberately CPU-only run (OVH's compute-optimized line has no
GPU; that's a separate OVH product), unlike every other stage in this
repo which runs on RunPod GPUs.

Region GRA9, not the default: the account's OVH quota is per-region, and
every other region this project has touched (WAW1) already has 6/6 cores
and 23/25 GB RAM used by two pre-existing, unrelated instances. GRA9 has
its own separate, completely unused 34-core/44GB quota — checked via
/cloud/project/{id}/quota before picking it, not assumed.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

BASE = "https://ca.api.ovh.com/1.0"
APP_KEY = os.environ["OVH_APP_KEY"]
APP_SECRET = os.environ["OVH_APP_SECRET"]
CONSUMER_KEY = os.environ["OVH_CONSUMER_KEY"]
PROJECT = os.environ.get("OVH_PROJECT_ID", "c8b896a127304d128835ef48da92ef4a")
REGION = os.environ.get("OVH_REGION", "GRA9")
FLAVOR_NAME = os.environ.get("OVH_FLAVOR", "c3-32")  # 16 vcpu, 32GB RAM
IMAGE_ID = os.environ.get("OVH_IMAGE_ID", "e65d6156-49cc-40ad-939d-0f7e0fa3e77f")  # Ubuntu 24.04, GRA9
SSH_KEY_NAME = "ai-train"  # already registered — comment brett-shaw@pldev, ~/.ssh/id_pldev_ci
INSTANCE_NAME = "rivaquant420b-quantal"


def _server_time() -> str:
    return urllib.request.urlopen(f"{BASE}/auth/time", timeout=10).read().decode().strip()


def ovh(method: str, path: str, body: dict | None = None) -> dict:
    url = BASE + path
    body_str = json.dumps(body) if body is not None else ""
    server_time = _server_time()
    sig_input = "+".join([APP_SECRET, CONSUMER_KEY, method, url, body_str, server_time])
    signature = "$1$" + hashlib.sha1(sig_input.encode()).hexdigest()
    headers = {
        "X-Ovh-Application": APP_KEY, "X-Ovh-Consumer": CONSUMER_KEY,
        "X-Ovh-Timestamp": server_time, "X-Ovh-Signature": signature,
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=body_str.encode() if body is not None else None,
                                  headers=headers, method=method)
    return json.load(urllib.request.urlopen(req, timeout=30))


def find_flavor_id() -> str:
    flavors = ovh("GET", f"/cloud/project/{PROJECT}/flavor?region={REGION}")
    for f in flavors:
        if f["name"] == FLAVOR_NAME:
            return f["id"]
    raise RuntimeError(f"flavor {FLAVOR_NAME} not found in {REGION}")


def find_sshkey_id() -> str:
    for k in ovh("GET", f"/cloud/project/{PROJECT}/sshkey"):
        if k["name"] == SSH_KEY_NAME:
            return k["id"]
    raise RuntimeError(f"ssh key {SSH_KEY_NAME!r} not found on this OVH project")


def create_instance() -> str:
    flavor_id = find_flavor_id()
    sshkey_id = find_sshkey_id()
    body = {
        "flavorId": flavor_id,
        "imageId": IMAGE_ID,
        "region": REGION,
        "name": INSTANCE_NAME,
        "sshKeyId": sshkey_id,
        "monthlyBilling": False,
    }
    resp = ovh("POST", f"/cloud/project/{PROJECT}/instance", body)
    return resp["id"]


def wait_for_ip(instance_id: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        inst = ovh("GET", f"/cloud/project/{PROJECT}/instance/{instance_id}")
        if inst.get("status") == "ACTIVE":
            for addr in inst.get("ipAddresses", []):
                if addr.get("type") == "public" and addr.get("version") == 4:
                    return addr["ip"]
        print(f"waiting for instance to boot (status={inst.get('status')})...", flush=True)
        time.sleep(10)
    raise TimeoutError(f"instance {instance_id} never got a public IP within {timeout_s}s")


def main() -> None:
    print(f"creating OVH instance ({FLAVOR_NAME}, {REGION}, compute-optimized, no GPU)...")
    instance_id = create_instance()
    print(f"instance id: {instance_id}")
    ip = wait_for_ip(instance_id)
    print(f"SSH ready (once cloud-init finishes): ssh -i ~/.ssh/id_pldev_ci root@{ip}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pod_info_ovh.json"), "w") as f:
        json.dump({"instance_id": instance_id, "ip": ip, "port": 22,
                   "flavor": FLAVOR_NAME, "region": REGION, "stage": "quantal"}, f)


if __name__ == "__main__":
    sys.exit(main())
