from modules.optim.adam import AdamOptim
from modules.optim.adabelief import AdaBeliefOptim
from modules.optim.scheduler import ScheduledOptim

optimizers = {"Adam": AdamOptim, "AdaBelief": AdaBeliefOptim}
