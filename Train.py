import json, time, os, sys, glob
import torch
from Common import *
from sklearn.model_selection import train_test_split
sys.path.insert(0, '..')
from COM_VAE import *
from torch.utils.data import DataLoader
from Featurize import frag_featurize
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')  
rdBase.DisableLog('rdApp.warning')  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_data = 'dataset.pkl'

criterion = Fragloss(pad=0)
epochs = 100
batch_size = 32
model = Space2Frag(use_gpu=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.5)

dataset = load_from_pickle(file_data)

train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
loader_validation = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print('Training:{}, Validation:{}, Test:{}'.format(len(train_data), len(validation_data), len(test_data)))

#——————————————————————————————————————————start——————————————————————————————————————————
log_name = ''
if log_name != '':
    base_folder = 'log/' + log_name + '/'
else:
    base_folder = time.strftime('log/%y%b%d_%I%M%p/', time.localtime())

if not os.path.exists(base_folder):
    os.makedirs(base_folder)
subfolders = ['checkpoints', 'plots']   
for subfolder in subfolders:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)

logfile = base_folder + 'log.txt'
with open(logfile, 'w') as f:
    f.write('Epoch\tTrain\tValidation\n')

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []
total_step = 0
patience = 10  
best_val_loss = float('inf')
epochs_without_improvement = 0

for e in range(epochs):
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(loader_train):
        start_batch = time.time()
        src, lengths, tgt, pdb_batch = frag_featurize(batch)
        elapsed_featurize = time.time() - start_batch
        optimizer.zero_grad()
        output, mu, sigma = model(src, lengths, pdb_batch)
        loss = criterion(output, tgt, mu, sigma)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        elapsed_batch = time.time() - start_batch
        elapsed_train = time.time() - start_train
        total_step += 1  
        print(total_step, elapsed_train, loss.item())

        train_sum += loss.cpu().data.numpy()
        train_weights += 1

    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            src, lengths, tgt, pdb_batch = frag_featurize(batch)
            output, mu, sigma = model(src, lengths, pdb_batch)
            loss = criterion(output, tgt, mu, sigma)

            validation_sum += loss.cpu().data.numpy()
            validation_weights += 1

    train_loss = train_sum / train_weights
    validation_loss = validation_sum / validation_weights
    print(f'Epoch {e+1}: Train Loss = {train_loss}, Validation Loss = {validation_loss}')

    with open(logfile, 'a') as f:
        f.write('{}\t{}\t{}\n'.format(e, train_loss, validation_loss))

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        epochs_without_improvement = 0
        checkpoint_filename = base_folder + 'checkpoints/epoch{}.pt'.format(e+1)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_filename)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {e+1} due to no improvement in validation loss.')
            break

    epoch_losses_valid.append(validation_loss)
    epoch_losses_train.append(train_loss)
    epoch_checkpoints.append(checkpoint_filename)
    scheduler.step(validation_loss)

# #_____________________________________continue——————————————————————————————————————————

checkpoint_path = 'log/24Dec19_0502PM/checkpoints/epoch29.pt'
logfile = 'log/24Dec19_0502PM/log.txt'
base_folder = 'log/24Dec19_0502PM/checkpoints/'

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  
else:
    print("No valid checkpoint found, starting from scratch.")
    start_epoch = 0  

log_name = time.strftime('%y%b%d_%I%M%p/', time.localtime())
base_folder = f'log/{log_name}/'
os.makedirs(base_folder, exist_ok=True)
os.makedirs(base_folder + 'checkpoints', exist_ok=True)
os.makedirs(base_folder + 'plots', exist_ok=True)

logfile = base_folder + 'log.txt'
with open(logfile, 'w') as f:
    f.write('Epoch\tTrain\tValidation\n')

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []
total_step = 0

for e in range(start_epoch, epochs):  
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(loader_train):
        start_batch = time.time()
        src, lengths, tgt, pdb_batch = frag_featurize(batch)
        elapsed_featurize = time.time() - start_batch
        
        optimizer.zero_grad()
        output, mu, sigma = model(src, lengths, pdb_batch)
        loss = criterion(output, tgt, mu, sigma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        elapsed_batch = time.time() - start_batch
        elapsed_train = time.time() - start_train
        total_step += 1
        print(total_step, elapsed_train, loss.item())

        train_sum += loss.cpu().data.numpy()
        train_weights += 1

    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            src, lengths, tgt, pdb_batch = frag_featurize(batch)
            output, mu, sigma = model(src, lengths, pdb_batch)
            loss = criterion(output, tgt, mu, sigma)

            validation_sum += loss.cpu().data.numpy()
            validation_weights += 1

    train_loss = train_sum / train_weights
    validation_loss = validation_sum / validation_weights
    print(f'Epoch {e+1}: Train Loss = {train_loss}, Validation Loss = {validation_loss}')

    with open(logfile, 'a') as f:
        f.write('{}\t{}\t{}\n'.format(e, train_loss, validation_loss))

    checkpoint_filename = base_folder + f'epoch{e+1}_step{total_step}.pt'
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_filename)

    epoch_losses_valid.append(validation_loss)
    epoch_losses_train.append(train_loss)
    epoch_checkpoints.append(checkpoint_filename)

