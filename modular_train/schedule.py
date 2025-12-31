class TrainingScheduler:
    def __init__(self, warmup_epochs=10, anneal_epochs=90):
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.anneal_end = warmup_epochs + anneal_epochs

    def get_temperature(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 2.0
        if epoch < self.anneal_end:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return 2.0 * (0.05 / 2.0) ** progress
        return 0.05

    def get_loss_weights(self, epoch: int):
        if epoch < self.warmup_epochs:
            return {
                "w_nll": 1.0,
                "w_final": 5.0,
                "w_aux_dist": 0.3,
                "w_aux_zone": 0.3,
                "w_gate": 1.0,
                "w_isolation": 3.0,
            }
        if epoch < self.anneal_end:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return {
                "w_nll": 1.0,
                "w_final": 5.0,
                "w_aux_dist": 0.3 + 0.2 * progress,
                "w_aux_zone": 0.3 + 0.2 * progress,
                "w_gate": 1.0 - 0.5 * progress,
                "w_isolation": 3.0,
            }
        return {
            "w_nll": 1.0,
            "w_final": 5.0,
            "w_aux_dist": 0.5,
            "w_aux_zone": 0.5,
            "w_gate": 0.5,
            "w_isolation": 3.0,
        }
