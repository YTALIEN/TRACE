from TRACE import *
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from dataset_generate import DataStreamGenerator
from collections  import Counter
from sklearn.utils.class_weight import compute_class_weight


def train_detection_network(train_loader,device,model_name='TRACE'):
    
    locate_time=time.localtime()
    model=TRACE().to(device)
    train_loss=[]
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion=nn.CrossEntropyLoss()
    epochs=args.epochs

    print(f'Training detection network:')
    
    time_info=f'{locate_time.tm_year}{locate_time.tm_mon}{locate_time.tm_mday}_{locate_time.tm_hour}{locate_time.tm_min}'
    writer=SummaryWriter(log_dir=f'./logs/{model_name}_{time_info}') # for tensorboard
    for epoch in range(epochs):
        train_progress_bar=tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Train-Epoch {epoch+1}/{epochs}',
            bar_format='{l_bar}{bar:20}{r_bar}'
        )
        model.train()
        train_loss_epoch=[]
        for _,(x,label,mask) in train_progress_bar:
            x=x.to(device)
            label=label.to(device,dtype=torch.long)  
            mask=mask.to(device)
            optimizer.zero_grad()
            output=model(x,mask)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            
            train_loss_epoch.append(loss.item())
            train_progress_bar.set_postfix({
            'Train_Loss': f'{loss.item():.4f}'
        })

        
        avg_train_loss=np.mean(train_loss_epoch)
        train_loss.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch+1)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')

        if (epoch+1)% 10 ==0 :
            checkpoint_dir=os.path.join('detector_network','checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_path=os.path.join(checkpoint_dir,
                                         f'detector_{model_name}_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'args': args  # opt: configuration parameters
            }, checkpoint_path)

    writer.close()
    # save the network
    directory = "detectors"  
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = f"{model_name}.pkl"    
    file_path = os.path.join(directory, file_name)  
    if not os.path.exists(directory):
        os.makedirs(directory) 
    torch.save(model.state_dict(),file_path)
    
    # save loss
    with open(f'detectors/{model_name}_loss.json', 'w') as f:
        json.dump({"train_loss": train_loss}, f)



def load_dataloader(benchmark,data_path=None):
    print(f'Loading data for {benchmark}...')
    if data_path is None:
        data_file=f'./data/window_seq_{benchmark}.pth'
    else:
        data_file=data_path
    if not os.path.exists(data_file):
        data_stream= DataStreamGenerator(benchmark=benchmark,args=args)
        all_data=data_stream.get_all_feature()
        file_directory = os.path.dirname(data_file)
        os.makedirs(file_directory, exist_ok=True)
        torch.save(all_data,data_file)
    balance_data_file=f'./data/window_seq_{benchmark}-balanced.pth'
    if not os.path.exists(balance_data_file):
        all_data=torch.load(data_file)
        _,count=get_weight(all_data)  
        balance_data=balance_class(all_data,count)
        torch.save(balance_data,f'./data/window_seq_{benchmark}-balanced.pth')
        get_weight(balance_data)  # print
    else:
        balance_data=torch.load(balance_data_file)
    
    train_loader=DataLoader(balance_data,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)
    return train_loader

def get_weight(all_data):
    all_labels = [int(label) for _, label in all_data]
    label_counts = Counter(all_labels)
    total = len(all_labels)
    print("Label distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"Class {label}: {count} samples ({100*count/total:.2f}%)")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights,label_counts

def balance_class(all_data,count):
    min_count=math.inf
    for label in count:
        if count[label]<min_count:
            min_count=count[label]
    for label in count:
        if count[label]>min_count:
            indices = [i for i, (_, l) in enumerate(all_data) if l == label]
            sampled_indices = np.random.choice(indices, min_count, replace=False)
            sampled_data = [all_data[i] for i in sampled_indices]
            all_data = [d for d in all_data if d[1] != label] + sampled_data
    return all_data


if __name__ == "__main__":
    benchmark='SDDObench' # your traning benchmark name
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # replace to your available device
    data_path=None # your data path, default is None, which will generate data for you
    train_loader=load_dataloader(benchmark=benchmark,data_path=data_path)
    model_name=f'TRACE-v{time.strftime("%Y%m%d-%H%M%S")}'
    train_detection_network(train_loader=train_loader,device=device,model_name=model_name)