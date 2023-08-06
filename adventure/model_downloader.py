from huggingface_hub import snapshot_download
import os


if __name__ == "__main__":
    model_path = "../models"
    repo_id = "TheBloke"
    model_name = "Llama-2-70B-chat-GPTQ"

    snapshot_download(
        repo_id=os.path.join(repo_id, model_name),
        local_dir=os.path.join(model_path, model_name)
    )
