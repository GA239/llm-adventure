from huggingface_hub import snapshot_download
import os

model_path = "../models"
repo_id = "bigscience"
model_name = "bloom"

snapshot_download(
    repo_id=os.path.join(repo_id, model_name),
    local_dir=os.path.join(model_path, model_name)
)
