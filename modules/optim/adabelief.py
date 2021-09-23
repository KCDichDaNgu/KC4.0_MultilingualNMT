import torch, math

class AdaBeliefOptim(torch.optim.Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    # No backward
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    # Initial step
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
            
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1  
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # This is the diff between Adam and Adabelief
                centered_grad = grad.sub(exp_avg)
                exp_avg_sq.mul_(beta2).addcmul_(centered_grad, centered_grad, value=1-beta2)
                # !

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
