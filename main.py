import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import os

import wandb

from models.absargsumm import AbsArgSumm
from utils import parser

if __name__ == "__main__":
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        print("Wandb API key not found")
        exit(1)
    wandb.login(key=api_key)

    args = parser()

    project_name = f"AbsArgSumm_{args.experiment}"
    if args.guided:
        project_name = f"Guided{project_name}"
    if args.shared_encoder:
        project_name = f"Shared{project_name}"
    os.environ["WANDB_PROJECT"] = project_name

    run = AbsArgSumm(
        experiment=args.experiment,
        guided=args.guided,
        shared_encoder=args.shared_encoder,
        project_name=project_name,
    )
    # run.train()
    # results = run.evaluate()
    # print(results)
    run.multirun()
