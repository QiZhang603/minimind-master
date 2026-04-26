import os


def init_tracker(args, ckp_data, run_name):
    """Initialize experiment tracker compatible with current checkpoint resume flow."""
    if not getattr(args, "use_wandb", False):
        return None

    tracker_name = getattr(args, "tracker", "swanlab")
    wandb_id = ckp_data.get("wandb_id") if ckp_data else None
    resume = "must" if wandb_id else None

    if tracker_name == "wandb":
        import wandb

        proxy = getattr(args, "wandb_proxy", None)
        if proxy:
            os.environ["HTTPS_PROXY"] = proxy
            os.environ["HTTP_PROXY"] = proxy
        entity = getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=wandb_id,
            resume=resume,
            entity=entity,
        )
        return wandb

    if tracker_name == "swanlab":
        import swanlab as wandb

        wandb.init(project=args.wandb_project, name=run_name, id=wandb_id, resume=resume)
        return wandb

    raise ValueError(f"Unsupported tracker backend: {tracker_name}")
