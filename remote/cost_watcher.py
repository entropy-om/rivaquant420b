"""Cost-safety watcher for the RivaQuant420b RunPod pod. Stops the pod
(releases the GPU, ends billing) the moment either the hard dollar cap is
hit or training reports done/error — never trust the training job to
clean up after itself, watch it from outside.
"""
import json
import os
import subprocess
import sys
import time

import requests

API = "https://api.runpod.io/graphql"
API_KEY = os.environ["RUNPOD_API_KEY"]
COST_CAP_USD = float(os.environ.get("RIVAQUANT_COST_CAP", "80"))
HOURLY_PRICE = float(os.environ.get("RIVAQUANT_HOURLY_PRICE", "0.34"))
POLL_SECS = int(os.environ.get("RIVAQUANT_WATCHER_POLL", "60"))
POD_INFO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pod_info.json")


def gql(query: str) -> dict:
    resp = requests.post(API, headers={"Authorization": f"Bearer {API_KEY}"},
                          json={"query": query}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]


def stop_pod(pod_id: str) -> None:
    gql(f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}')


def remote_status(ip: str, port: int) -> dict | None:
    try:
        out = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(port), f"root@{ip}", "cat /workspace/rivaquant420b-out/status.json"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            return json.loads(out.stdout)
    except Exception as error:
        print(f"status check failed (non-fatal): {error}")
    return None


def main() -> None:
    with open(POD_INFO_PATH) as f:
        info = json.load(f)
    pod_id, ip, port = info["pod_id"], info["ip"], info["port"]

    started = time.time()
    print(f"watching pod {pod_id} (stage={info.get('stage', '?')}) — cap ${COST_CAP_USD:.2f} "
          f"at ${HOURLY_PRICE}/hr (~{COST_CAP_USD / HOURLY_PRICE:.1f}h budget)")

    while True:
        elapsed_hours = (time.time() - started) / 3600
        cost_so_far = elapsed_hours * HOURLY_PRICE
        print(f"elapsed {elapsed_hours:.2f}h  est. cost ${cost_so_far:.2f}")

        if cost_so_far >= COST_CAP_USD:
            print(f"COST CAP HIT (${cost_so_far:.2f} >= ${COST_CAP_USD:.2f}) — stopping pod")
            stop_pod(pod_id)
            return

        status = remote_status(ip, port)
        if status and status.get("stage") in ("done", "error"):
            print(f"training reported stage={status['stage']} — stopping pod")
            stop_pod(pod_id)
            return

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    sys.exit(main())
