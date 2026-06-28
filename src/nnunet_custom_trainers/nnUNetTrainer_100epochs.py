from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_100epochs(nnUNetTrainer):
    """
    Controlled-budget nnU-Net v2 trainer.

    Only changes training length from the default 1000 epochs to 100 epochs.
    Architecture, loss, augmentation, optimizer, scheduler, and inference remain nnU-Net defaults.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        self.num_epochs = 100