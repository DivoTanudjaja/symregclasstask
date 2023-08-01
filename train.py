import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_precompute import PrecomputedDataset
from model import GPTConfig, GPT

import matplotlib.pyplot as plt
import numpy as np

# Model arguments

n_layer = 6 
n_head = 6
n_embd = 384
block_size = 128
bias = False
dropout = 0.0

device = 'cuda'
dtype = 'float16' 

# Model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=14, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)
model = torch.compile(model)

# Hyperparameters
batch_size = 512
epochs = 2

# DataLoaders
train_dataloader = DataLoader(PrecomputedDataset('data/nesymres/train_nc.pt'), batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(PrecomputedDataset('data/nesymres/val_nc.pt'), batch_size, shuffle=False, drop_last=True)

# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = epochs*len(train_dataloader) # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 3*len(train_dataloader) # how many steps to warm up for
lr_decay_iters = epochs*len(train_dataloader) # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
checkpoint = None # free up memory

# Accuracy and loss lists
train_complete_accuracy_list = []
test_complete_accuracy_list = []
train_avg_accuracy_list = []
test_avg_accuracy_list = []
train_partial_accuracy_list = []
test_partial_accuracy_list = []
train_loss_list = []
test_loss_list = []
train_fixed_avg_accuracy_list = []
test_fixed_avg_accuracy_list = []
learning_rate_list = []


# Learning rate decay
current_iteration = 0
def get_lr(it):
    if not decay_lr:
        return learning_rate
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Linear decay down to min_lr after warmup
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    decay_ratio = max(0.0, min(1.0, decay_ratio))  # Ensure the ratio is within 0 and 1
    return learning_rate - decay_ratio * (learning_rate - min_lr)

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    global current_iteration
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.float32, non_blocking=True)

        logits, loss = model(X, y)

        # backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Compute the current learning rate based on the current_iteration and update optimizer
        current_lr = get_lr(current_iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        learning_rate_list.append(current_lr)

        current_iteration = current_iteration + 1


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, correct_less_strict = 0, 0, 0

    index_accuracies = [0] * len(dataloader.dataset[0][1])

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            logits, loss = model(X, y)
            pred = (torch.sigmoid(logits) > 0.5).type(torch.float16)


            test_loss += loss.item()

            for i in range(len(index_accuracies)):
                index_accuracies[i] += (pred[:, i] == y[:, i]).sum().type(torch.float16).sum().item()
            # multihot prediction in pred, shape is (batch_size, 14)
            # multihot ground truth in y, shape is (batch_size, 14)
            # correct only if all is correct
            correct += (pred == y).all(dim=1).type(torch.float16).sum().item()
            # accuracy without the strict requirement that all predictions need to be correct
            correct_less_strict += (pred == y).sum(dim=1).type(torch.float16).sum().item()

    test_loss /= num_batches
    correct /= size
    partial_correct = correct_less_strict / size
    partial_correct /= 14
    print("----")
    print(size)
    print("----")
    avg_accuracies = [index_accuracies[i] * 100 / size for i in range(len(index_accuracies))]

    if dataloader == train_dataloader:
        print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        train_complete_accuracy_list.append(100*correct)
        train_loss_list.append(test_loss)
        train_partial_accuracy_list.append(partial_correct*100)
        train_avg_accuracy_list.append(avg_accuracies)
    else:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        test_complete_accuracy_list.append(100*correct)
        test_loss_list.append(test_loss)
        test_partial_accuracy_list.append(partial_correct*100)
        test_avg_accuracy_list.append(avg_accuracies)

if __name__ == '__main__':
    print(f"Epoch 0\n-------------------------------")
    test_loop(val_dataloader, model)
    test_loop(train_dataloader, model)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(val_dataloader, model)
        test_loop(train_dataloader, model)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, 'model_checkpoint_'+str(learning_rate)+'.pt')

    # Combine the lists into a dictionary for easier saving
    data_dict = {
        'train_complete_accuracy': train_complete_accuracy_list,
        'test_complete_accuracy': test_complete_accuracy_list,
        'train_loss': train_loss_list,
        'test_loss': test_loss_list,
        'learning_rate': learning_rate_list,
        'train_partial_accuracy': train_partial_accuracy_list,
        'test_partial_accuracy': test_partial_accuracy_list,
        'train_avg_accuracy': train_fixed_avg_accuracy_list,
        'test_avg_accuracy': test_fixed_avg_accuracy_list,
    }

    # Define the file path where you want to save the data
    save_path = 'saved_data_'+str(learning_rate)+'.pt'

    # Save the data using torch.save()
    torch.save(data_dict, save_path)

    plt.figure(figsize=(10, 5))
    epochs_range = range(epochs + 1)

    # Complete accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_complete_accuracy_list, label='Complete Accuracy - Train', marker='o', c='blue')
    plt.plot(epochs_range, test_complete_accuracy_list, label='Complete Accuracy - Test', marker='o',  c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Complete Accuracy (%)')
    plt.title('Complete Accuracy')
    plt.legend()

    # Partial accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_partial_accuracy_list, label='Partial Accuracy - Train', marker='o', c='blue')
    plt.plot(epochs_range, test_partial_accuracy_list, label='Partial Accuracy - Test', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Partial Accuracy (%)')
    plt.title('Partial Accuracy')
    plt.legend()

    # Training loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, train_loss_list, label='Train Loss', marker='o', c='blue')
    plt.plot(epochs_range, test_loss_list, label='Test Loss', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Learning rate
    plt.subplot(2, 2, 4)
    plt.plot(range(current_iteration), learning_rate_list, label='Learning Rate', marker='o', c='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig('results_'+str(learning_rate)+'.png', dpi=1000)

    train_fixed_avg_accuracy_list = [[sublist[i] for sublist in train_avg_accuracy_list] for i in range(len(train_avg_accuracy_list[0]))]
    test_fixed_avg_accuracy_list = [[sublist[i] for sublist in test_avg_accuracy_list] for i in range(len(test_avg_accuracy_list[0]))]
    
    plt.figure(figsize=(10, 20))
    epochs_range = range(epochs + 1)
    
    # Training accuracy
    plt.subplot(7, 2, 1)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[0], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[0], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (+)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 2)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[1], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[1], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (×)')
    plt.legend() 
    plt.ylim(0, 1)   

    plt.subplot(7, 2, 3)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[2], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[2], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (-)')
    plt.legend() 
    plt.ylim(0, 1) 

    plt.subplot(7, 2, 4)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[3], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[3], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (÷)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 5)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[4], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[4], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (√)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 6)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[5], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[5], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (^2)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 7)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[6], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[6], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (^3)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 8)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[7], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[7], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (^4)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 9)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[8], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[8], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (log)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 10)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[9], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[9], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (exp)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 11)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[10], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[10], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (sin)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 12)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[11], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[11], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (cos)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 13)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[12], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[12], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (tan)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplot(7, 2, 14)
    plt.plot(epochs_range, train_fixed_avg_accuracy_list[13], label='Average Training Accuracy', marker='o', c='blue')
    plt.plot(epochs_range, test_fixed_avg_accuracy_list[13], label='Average Test Accuracy', marker='o', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy (arcsin)')
    plt.legend()
    plt.ylim(0, 1)

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.subplots_adjust(wspace=0.3)  # Adjust the horizontal spacing between subplots

    plt.savefig('partial_results_'+str(learning_rate)+'.png', dpi=1000)
    
    print("Done!")
