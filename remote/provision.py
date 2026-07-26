"""Provision a single RunPod GPU pod for a RivaQuant420b training stage.

Stage 1 (162m, the pipeline's own regression test) needs no more than the
original 162M run did: RTX 4090 community cloud (~$0.34/hr) is plenty of
VRAM and cheap enough that a real cost cap buys hundreds of GPU-hours.
Larger stages (1b/8b/70b/420b) will need bigger cards (H200) and real
parallelism (FSDP/TP/PP) sized to what's actually booked at that stage —
not provisioned speculatively here before that stage is reached.
"""
import json
import os
import sys
import time

import requests

API = "https://api.runpod.io/graphql"
API_KEY = os.environ["RUNPOD_API_KEY"]
STAGE = os.environ.get("RIVAQUANT_STAGE", "162m")
GPU_TYPE_ID = os.environ.get("RIVAQUANT_GPU_TYPE", "NVIDIA GeForce RTX 4090")
IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
SSH_PUBKEY_PATH = os.path.expanduser("~/.ssh/id_ed25519.pub")
POD_NAME = f"rivaquant420b-{STAGE}"


def gql(query: str) -> dict:
    resp = requests.post(API, headers={"Authorization": f"Bearer {API_KEY}"},
                          json={"query": query}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]


def create_pod() -> str:
    with open(SSH_PUBKEY_PATH) as f:
        pubkey = f.read().strip()

    # Deploying via a raw imageName (not a RunPod "template" object) skips
    # whatever startup wrapper their own templates use to auto-configure
    # sshd — confirmed by hand on the original rivaquant run: the vanilla
    # runpod/pytorch image never opened port 22 without this.
    start_cmd = (
        "bash -c '"
        "apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server; "
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh; "
        f"echo \\\"{pubkey}\\\" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys; "
        "service ssh start; "
        "sleep infinity'"
    )

    query = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: COMMUNITY
        gpuCount: 1
        volumeInGb: 30
        containerDiskInGb: 20
        minVcpuCount: 4
        minMemoryInGb: 16
        gpuTypeId: "{GPU_TYPE_ID}"
        name: "{POD_NAME}"
        imageName: "{IMAGE}"
        dockerArgs: "{start_cmd}"
        ports: "22/tcp"
        volumeMountPath: "/workspace"
      }}) {{ id }}
    }}
    """
    data = gql(query)
    return data["podFindAndDeployOnDemand"]["id"]


def wait_for_ssh(pod_id: str, timeout_s: int = 300) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = gql(f'query {{ pod(input: {{podId: "{pod_id}"}}) {{ runtime {{ ports {{ ip isIpPublic privatePort publicPort type }} }} }} }}')
        runtime = data["pod"]["runtime"]
        if runtime:
            for p in runtime["ports"]:
                if p["privatePort"] == 22 and p["isIpPublic"]:
                    return p["ip"], p["publicPort"]
        print("waiting for pod to boot + expose SSH...", flush=True)
        time.sleep(10)
    raise TimeoutError(f"pod {pod_id} never exposed SSH within {timeout_s}s")


def main() -> None:
    print(f"creating pod ({GPU_TYPE_ID}, community cloud, stage={STAGE})...")
    pod_id = create_pod()
    print(f"pod id: {pod_id}")
    ip, port = wait_for_ssh(pod_id)
    print(f"SSH ready: ssh -p {port} root@{ip}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pod_info.json"), "w") as f:
        json.dump({"pod_id": pod_id, "ip": ip, "port": port, "gpu_type": GPU_TYPE_ID, "stage": STAGE}, f)


if __name__ == "__main__":
    sys.exit(main())
