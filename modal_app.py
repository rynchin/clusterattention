import modal

app = modal.App("ClusterAttention_layers8")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "fastapi",
        "uvicorn",
    )
    .add_local_dir(".", "/root/project")
)

volume = modal.Volume.from_name("enwik8-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
)

@modal.web_endpoint()
def run_training():
    import os
    import sys

    sys.path.append("/root/project")

    enwik8_path = "/data/enwik8"
    if not os.path.exists(enwik8_path):
        raise FileNotFoundError(
            f"Expected {enwik8_path}. Upload enwik8 into the 'enwik8-data' volume."
        )

    os.environ["ENWIK8_PATH"] = enwik8_path

    from train import run_all_models

    results = run_all_models()
    print("Final results:", results)
    return results

@app.local_entrypoint()
def main():
    res = run_training.remote()
    print("Remote results:", res)

@app.function(
    image=image,
    volumes={"/data": volume},
)
def upload_enwik8_from_local():
    import shutil, os

    src = "/root/project/enwik8"
    dst = "/data/enwik8"

    if not os.path.exists(src):
        raise FileNotFoundError(f"Local file {src} not found")

    shutil.copy(src, dst)
    print(f"Copied {src} -> {dst}")