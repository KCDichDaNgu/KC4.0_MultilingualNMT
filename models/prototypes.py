import torch
import torch.nn as nn
import copy, time

from modules import *
from utils.data import create_fields, create_dataset, read_data
from utils.loss import LabelSmoothingLoss
from utils.metric import bleu
from utils.save import save_model_to_path, load_model_from_path, check_model_in_path
from utils.misc import create_masks

class Transformer(nn.Module):
    """ Cuối cùng ghép chúng lại với nhau để được mô hình transformer hoàn chỉnh
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = Config() if(config is None) else Config(config)
        opt = self.config.opt
        # input fields
        self.SRC, self.TRG = create_fields(opt['src_lang'], opt['trg_lang'])
        # initialize dataset and by proxy the vocabulary
        train_src_data, train_trg_data = read_data(opt['train_src_data'], opt['train_trg_data'])
        self.raw_valid_data = valid_src_data, valid_trg_data = read_data(opt['valid_src_data'], opt['valid_trg_data'])
        self.train_iter = create_dataset(train_src_data, train_trg_data, opt['max_strlen'], opt['batchsize'], opt['device'], self.SRC, self.TRG, istrain=True)
        self.valid_iter = create_dataset(valid_src_data, valid_trg_data, opt['max_strlen'], opt['batchsize'], opt['device'], self.SRC, self.TRG, istrain=False)
        src_vocab, trg_vocab = len(self.SRC.vocab), len(self.TRG.vocab)
        # vocab
        d_model, N, heads, dropout = opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        """
        src: batch_size x seq_length
        trg: batch_size x seq_length
        src_mask: batch_size x 1 x seq_length
        trg_mask batch_size x 1 x seq_length
        output: batch_size x seq_length x vocab_size
        """
        e_outputs = self.encoder(src, src_mask)
        
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
 
    def train_step(self, optimizer,batch, criterion):
        """
        Một lần cập nhật mô hình
        """
        self.train()
        opt = self.config.opt
        
        src = batch.src.transpose(0,1).cuda()
        trg = batch.trg.transpose(0,1).cuda()
        trg_input = trg[:, :-1]
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
        src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, opt['device'])
        preds = self(src, trg_input, src_mask, trg_mask)
    
        ys = trg[:, 1:].contiguous().view(-1)
    
        optimizer.zero_grad()
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        loss.backward()
        optimizer.step_and_update_lr()
        
        loss = loss.item()
        
        return loss    

    def validate(self, valid_iter, criterion):
        """ Tính loss trên tập validation
        """
        self.eval()
        opt = self.config.opt
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
    
        with torch.no_grad():
            total_loss = []
            for batch in valid_iter:
                src = batch.src.transpose(0,1).cuda()
                trg = batch.trg.transpose(0,1).cuda()
                trg_input = trg[:, :-1]
                src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, opt['device'])
                preds = self(src, trg_input, src_mask, trg_mask)
    
                ys = trg[:, 1:].contiguous().view(-1)
    
                loss = criterion(preds.view(-1, preds.size(-1)), ys)
    
                loss = loss.item()
    
                total_loss.append(loss)
    
        avg_loss = np.mean(total_loss)
    
        return avg_loss

    def translate_sentence(self, sentence, device, k, max_len):
        self.eval()
        return translate_sentence(sentence, self, self.SRC, self.TRG, device, k, max_len)

    def run_train(self, model_dir=None, config=None):
        opt = self.config.opt
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
        
        model = self
    
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        model = model.to(opt['device'])
        if(model_dir is not None):
            checkpoint_idx = check_model_in_path(model_dir)
            if(checkpoint_idx > 0):
                print("Found model with index {:d} already saved.".format(checkpoint_idx))
                load_model_from_path(model, model_dir, checkpoint_idx=checkpoint_idx)
        else:
            checkpoint_idx = 0
        
        optimizer = ScheduledOptim(
                torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                0.2, opt['d_model'], 4000)
        
        criterion = LabelSmoothingLoss(len(self.TRG.vocab), padding_idx=trg_pad, smoothing=0.1)

        valid_src_data, valid_trg_data = self.raw_valid_data
        # debug
        bleuscore = bleu(valid_src_data[:10], valid_trg_data[:10], model, opt['device'], opt['k'], opt['max_strlen'])
        print("Initialization bleuscore: {:.4f}".format(bleuscore))

        for epoch in range(checkpoint_idx, opt['epochs']):
            total_loss = 0
            
            for i, batch in enumerate(self.train_iter): 
                s = time.time()
                loss = self.train_step(optimizer, batch, criterion)
                
                total_loss += loss
                
                if (i + 1) % opt['printevery'] == 0:
                    avg_loss = total_loss/opt['printevery']
                    print('epoch: {:03d} - iter: {:05d} - train loss: {:.4f} - time: {:.4f}'.format(epoch, i, avg_loss, time.time()- s))
                    total_loss = 0
        
            s = time.time()
            valid_loss = self.validate(self.valid_iter, criterion)
            bleuscore = bleu(valid_src_data[:500], valid_trg_data[:500], model, opt['device'], opt['k'], opt['max_strlen'])
            print('epoch: {:03d} - iter: {:05d} - valid loss: {:.4f} - bleu score: {:.4f} - time: {:.4f}'.format(epoch, i, valid_loss, bleuscore, time.time() - s))
            if(model_dir is not None):
                save_model_to_path(model, model_dir, checkpoint_idx=epoch+1)

    def run_infer(self, inputs):
        opt = self.config.opt
        results = [ translate_sentence(sentence, self, self.SRC, self.TRG, opt['device'], opt['k'], opt['max_strlen']) for sentence in inputs ]
        return results

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def state_dict(self):
        optimizer_state_dict = {
            'init_lr':self.init_lr,
            'd_model':self.d_model,
            'n_warmup_steps':self.n_warmup_steps,
            'n_steps':self.n_steps,
            '_optimizer':self._optimizer.state_dict(),
        }
        
        return optimizer_state_dict
    
    def load_state_dict(self, state_dict):
        self.init_lr = state_dict['init_lr']
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_steps = state_dict['n_steps']
        
        self._optimizer.load_state_dict(state_dict['_optimizer'])
        
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
