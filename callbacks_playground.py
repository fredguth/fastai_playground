"Tools to help find the optimal learning rate for training"
from ..torch_core import *
from ..data import DataBunch
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['LRFinder']

class LRFinder(LearnerCallback):
    """Causes `learn` to go on a mock training from `start_lr` to `end_lr` for `num_it` iterations.
       Training is interrupted if the loss diverges. Weights changes are reverted after run complete."""
    def __init__(self, learn:Learner, start_lr:float=1e-5, end_lr:float=10, num_it:int=100):
        "Initialize schedule of learning rates"
        super().__init__(learn)
        self.data = learn.data
        self.sched = Stepper((start_lr, end_lr), num_it, annealing_exp)
        #To avoid validating if the train_dl has less than num_it batches, we put aside the valid_dl and remove it
        #during the call to fit.
        self.valid_dl = learn.data.valid_dl
        self.data.valid_dl = None

    def on_train_begin(self, **kwargs:Any)->None:
        "Initialize optimizer and learner hyperparameters."
        self.learn.save('tmp')
        self.opt = self.learn.opt
        self.opt.lr = self.sched.start
        self.stop,self.best_loss = False,0.

    def on_batch_end(self, iteration:int, smooth_loss:TensorOrNumber, **kwargs:Any)->None:
        "Determine if loss has runaway and we should stop."
        if iteration==0 or smooth_loss < self.best_loss: self.best_loss = smooth_loss
        self.opt.lr = self.sched.step()
        if self.sched.is_done or smooth_loss > 4*self.best_loss:
            #We use the smoothed loss to decide on the stopping since it's less shaky.
            self.stop=True
            return True

    def on_epoch_end(self, **kwargs:Any)->None:
        "Tell Learner if we need to stop."
        return self.stop

    def on_train_end(self, **kwargs:Any)->None:
        "Cleanup learn model weights disturbed during LRFind exploration."
        # restore the valid_dl we turned of on `__init__`
        self.data.valid_dl = self.valid_dl
        self.learn.load('tmp')
        if hasattr(self.learn.model, 'reset'): self.learn.model.reset()
class Recorder(LearnerCallback):
    "A `LearnerCallback` that records epoch, loss, opt and metric data during training."
    _order=-10
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.opt = self.learn.opt
        self.train_dl = self.learn.data.train_dl

    def on_train_begin(self, pbar:PBar, metrics:MetricFuncList, **kwargs:Any)->None:
        "Initialize recording status at beginning of training."
        self.pbar = pbar
        self.names = ['epoch', 'train loss', 'valid loss'] + [fn.__name__ for fn in metrics]
        if hasattr(self, '_added_met_names'): self.names += self._added_met_names
        self.pbar.write('  '.join(self.names))
        self.losses,self.val_losses,self.lrs,self.moms,self.metrics,self.nb_batches = [],[],[],[],[],[]

    def on_batch_begin(self, train, **kwargs:Any)->None:
        "Record learning rate and momentum at beginning of batch."
        if train:
            self.lrs.append(self.opt.lr)
            self.moms.append(self.opt.mom)

    def on_backward_begin(self, smooth_loss:Tensor, **kwargs:Any)->None:
        "Record the loss before any other callback has a chance to modify it."
        self.losses.append(smooth_loss)
        if self.pbar is not None and hasattr(self.pbar,'child'):
            self.pbar.child.comment = f'{smooth_loss:.4f}'

    def on_epoch_end(self, epoch:int, num_batch:int, smooth_loss:Tensor,
                     last_metrics=MetricsList, **kwargs:Any)->bool:
        "Save epoch info: num_batch, smooth_loss, metrics."
        self.nb_batches.append(num_batch)
        if last_metrics is not None:
            self.val_losses.append(last_metrics[0])
            if hasattr(self, '_added_mets'): last_metrics += self._added_mets
            if len(last_metrics) > 1: self.metrics.append(last_metrics[1:])
            self.format_stats([epoch, smooth_loss] + last_metrics)
        else:  self.format_stats([epoch, smooth_loss])
        return False

    def format_stats(self, stats:TensorOrNumList)->None:
        "Format stats before printing."
        str_stats = []
        for name,stat in zip(self.names,stats):
            t = str(stat) if isinstance(stat, int) else f'{stat:.6f}'
            t += ' ' * (len(name) - len(t))
            str_stats.append(t)
        self.pbar.write('  '.join(str_stats))
        
    def add_metrics(self, metrics):
        self._added_mets = metrics

    def add_metric_names(self, names):
        self._added_met_names = names
        
    def plot_lr(self, show_moms=False)->None:
        "Plot learning rate, `show_moms` to include momentum."
        iterations = range_of(self.lrs)
        if show_moms:
            _, axs = plt.subplots(1,2, figsize=(12,4))
            axs[0].plot(iterations, self.lrs)
            axs[1].plot(iterations, self.moms)
        else: plt.plot(iterations, self.lrs)

    def plot(self, skip_start:int=10, skip_end:int=5)->None:
        "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`."
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        _, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_xscale('log')

    def plot_losses(self)->None:
        "Plot training and validation losses."
        _, ax = plt.subplots(1,1)
        iterations = range_of(self.losses)
        ax.plot(iterations, self.losses)
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        ax.plot(val_iter, self.val_losses)

    def plot_metrics(self)->None:
        "Plot metrics collected during training."
        assert len(self.metrics) != 0, "There are no metrics to plot."
        _, axes = plt.subplots(len(self.metrics[0]),1,figsize=(6, 4*len(self.metrics[0])))
        val_iter = self.nb_batches
        val_iter = np.cumsum(val_iter)
        axes = axes.flatten() if len(self.metrics[0]) != 1 else [axes]
        for i, ax in enumerate(axes):
            values = [met[i] for met in self.metrics]
            ax.plot(val_iter, values)
