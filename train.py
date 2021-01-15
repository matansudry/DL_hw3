import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger
import torch.optim as optim
import IPython.display
import numpy as np
import utils.plot as plot
import matplotlib.pyplot as plt

def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(dis_model: nn.Module, gen_model: nn.Module, train_loader: DataLoader, train_params: TrainParams, logger: TrainLogger) -> Metrics:
    dsc_avg_losses, gen_avg_losses = [], []
    dsc_optimizer = torch.optim.Adam(dis_model.parameters(), lr=train_params.lr_des)
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=train_params.lr_gen)
    for epoch_idx in range(train_params.num_epochs):
        # We'll accumulate batch losses and show an average once per epoch.
        dsc_losses, gen_losses = [], []
        print(f'--- EPOCH {epoch_idx+1}/{train_params.num_epochs} ---')

        for x_data in train_loader:
            image = x_data[0]
            features = x_data[1]
            if torch.cuda.is_available():
                image = image.to("cuda")
                features = features.to("cuda")
            dsc_loss, gen_loss = train_batch(dis_model, gen_model, discriminator_loss_fn, generator_loss_fn, dsc_optimizer, gen_optimizer, image, features)
            dsc_losses.append(dsc_loss)
            gen_losses.append(gen_loss)

        dsc_avg_losses.append(np.mean(dsc_losses))
        gen_avg_losses.append(np.mean(gen_losses))
        print(f'Discriminator loss: {dsc_avg_losses[-1]}')
        print(f'Generator loss:     {gen_avg_losses[-1]}')
        
        if save_checkpoint(gen_model, dsc_avg_losses, gen_avg_losses, 'checkpoint'):
            print(f'Saved checkpoint.')
            

def discriminator_loss_fn(y_data, y_generated, data_label=1, label_noise=0.3):
    assert data_label == 1 or data_label == 0
    device = y_generated.device
    norm_label_noise = 0.5 * label_noise

    min_value, max_value =  [data_label-norm_label_noise, data_label+norm_label_noise]
    diff = max_value-min_value
    noisy_data_label = min_value + torch.rand(y_data.shape, device=device)*diff

    min_value, max_value =  [1-data_label-norm_label_noise, 1-data_label+norm_label_noise]
    diff = max_value - min_value
    generated_label = min_value + torch.rand(y_generated.shape, device=device)*diff
    
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    loss_data = loss_func(y_data, noisy_data_label)
    loss_generated = loss_func(y_generated, generated_label)
    return loss_data + loss_generated

def generator_loss_fn(y_generated, data_label=1):
    assert data_label == 1 or data_label == 0
    device = y_generated.device
    generated_labeles = torch.ones(y_generated.shape, device=device)
    generated_labeles = data_label * generated_labeles
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss = loss_func(y_generated, generated_labeles)
    return loss

def create_optimizer(model_params, opt_params):
    opt_params = opt_params.copy()
    optimizer_type = opt_params['type']
    opt_params.pop('type')
    return optim.__dict__[optimizer_type](model_params, **opt_params)

def train_batch(
    dsc_model,
    gen_model,
    dsc_loss_fn,
    gen_loss_fn,
    dsc_optimizer,
    gen_optimizer,
    image,
    features,
):
    dsc_optimizer.zero_grad()
    
    #real images
    y_data = dsc_model.forward(image)
    
    #generate data w/o grad
    generated_data = gen_model.sample(image.shape[0], features, with_grad=False)

    #fake images w/o grad
    y_generated = dsc_model.forward(generated_data.detach())

    #fix y shape
    y_data = y_data.view(-1)
    y_generated = y_generated.view(-1)
    
    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward(retain_graph=True)
    dsc_optimizer.step()

    gen_optimizer.zero_grad()
    
    #generate data w/ grad
    generated_data_2 = gen_model.sample(image.shape[0], features, with_grad=True)

    #fake images w/ grad
    y_generated_2 = dsc_model(generated_data_2)

    #fix y shape
    y_generated = y_generated.view(-1)
    
    gen_loss = gen_loss_fn(y_generated_2.view(-1)) 
    gen_loss.backward()
    gen_optimizer.step()

    return dsc_loss.item(), gen_loss.item()

def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    saved = False
    gen_len = len(gen_losses)
    checkpoint_file = f"output/{checkpoint_file}"+str(gen_len)+".pt"
    torch.save(gen_model, checkpoint_file)
    saved = True
    return saved
