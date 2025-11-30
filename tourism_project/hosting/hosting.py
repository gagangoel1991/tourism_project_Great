from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("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"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="ggoel1991/tourism_project",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
