PAPER_DATASETS = [
    "ncRNADrug",
    "openTargets",
    "CTD",
    "PrimeKG",
]

RELEASE_LAMBDA_BR = 0.008
RELEASE_LAMBDA_PERT = 0.06


UNIFIED_HYPERPARAMS = {
    "hidden_dim": 128,
    "lr": 0.001,
    "out_channels": 256,
    "dropout1": 0.1,
    "bridge_topk": 8,
    "subgraph_hops": 2,
    "bridge_prior_strength": 1.0,
    "weight_decay": 5e-5,
    "lambda_br": RELEASE_LAMBDA_BR,
    "lambda_pert": RELEASE_LAMBDA_PERT,
    "latent_mediator_count": 16,
    "mediator_dropout": 0.2,
}


def apply_release_defaults(args):
    args.link_batch_size = 1024
    args.grad_clip = 1.0
    args.utility_gate_floor = 0.50
    args.threshold_metric = "mcc"
    args.threshold_strategy = "validation_auto"

    for key, value in UNIFIED_HYPERPARAMS.items():
        setattr(args, key, value)
    return args
