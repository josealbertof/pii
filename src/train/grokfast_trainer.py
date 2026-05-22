from transformers import Trainer
from train.grokfast import gradfilter_ema, gradfilter_ma

class GrokfastTrainer(Trainer):
    def __init__(self, *args, grokfast_type="ema", grokfast_alpha=0.98, grokfast_lamb=2.0,
                 grokfast_window_size=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.grokfast_type = grokfast_type
        self.grokfast_alpha = grokfast_alpha
        self.grokfast_lamb = grokfast_lamb
        self.grokfast_window_size = grokfast_window_size
        self._grads = None  # Running gradient memory

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Overrides Trainer.training_step to inject the Grokfast gradient filter
        between loss.backward() and optimizer.step().
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Scale loss for gradient accumulation BEFORE backward
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # ── Grokfast: inject gradient filter here ──────────────────────────
        # Only apply on the last accumulation step (when optimizer.step() will be called)
        is_last_accumulation_step = (
            self.state.global_step % self.args.gradient_accumulation_steps == 0
            if self.args.gradient_accumulation_steps > 1
            else True
        )
        if is_last_accumulation_step:
            if self.grokfast_type == "ema":
                self._grads = gradfilter_ema(
                    model,
                    grads=self._grads,
                    alpha=self.grokfast_alpha,
                    lamb=self.grokfast_lamb,
                )
            elif self.grokfast_type == "ma":
                self._grads = gradfilter_ma(
                    model,
                    grads=self._grads,
                    window_size=self.grokfast_window_size,
                    lamb=self.grokfast_lamb,
                )
        # ───────────────────────────────────────────────────────────────────

        return loss.detach()