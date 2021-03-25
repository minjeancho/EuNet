import torch 
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from model import MLP, LSTM, FeedbackModel, EuNet
from preprocess import SongDataset


def play_notes(my_piano, note_array, tempo=0.3):
    for t in range(note_array.shape[0]):
        note = int(note_array[t])
        my_piano[note-21].play()
        time.sleep(tempo)

def train(model, data_loader, model_name, window_size = 4):
    print("training %s ..." %model_name)
    loss_all = []
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    hidden_sz = hyperparams["hidden_sz"]
    vocab_sz =  hyperparams["vocab_sz"]
    all_sz = window_size + hidden_sz + hidden_sz + vocab_sz

    for epoch in tqdm(range(hyperparams["num_epochs"])):
        for batch in data_loader:

            if model_name == "mlp":
                x = batch["train_input"].float()
                x_label = batch['train_label']
                x_pred = model(x)

            elif model_name == "lstm":
                x = batch['train_input'].long()
                x_label = batch['train_label']
                x_pred = model(x)
                x_pred = torch.reshape(x_pred, (x_pred.size()[0] * x_pred.size()[1], x_pred.size()[2]))
            
            elif model_name == "feedback":
                x = batch["train_input"].float()
                x_label = batch['train_label']

                if epoch == 0:
                    # x_memory initialized as 0's
                    x_memory = torch.zeros(x.size()[0], hidden_sz)
                    x_memory, x_pred = model(x, x_memory)
                    # detach memory from computation graph 
                    x_memory = x_memory.detach()
                else:
                    x_memory, x_pred = model(x, x_memory)
                    # detach memory from computation graph 
                    x_memory = x_memory.detach()

            elif model_name == "eunet":
                x = batch["train_input"].float()
                x_label = batch['train_label']

                if epoch == 0:
                    # x_all initialized as 0's
                    x_all = torch.zeros(x.size()[0], all_sz)
                    x0, x1, x2, x_pred  = model(x, x_all)
                    # detach from computation graph 
                    x_all = torch.cat((x0, x1, x2, x_pred), -1).detach()
                else:
                    x0, x1, x2, x_pred  = model(x, x_all)
                    # detach from computation graph 
                    x_all = torch.cat((x0, x1, x2, x_pred), -1).detach()
                
            loss = loss_fn(x_pred, x_label.long())
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #plt.plot(loss_all) 
    #plt.show()
    print("avg %s data loss: %f" %(model_name, np.mean(loss_all)))

def test(model, model_name, window_size = 4):
    print("trained %s model playing music..." %model_name)
    # max time step 
    t_max = hyperparams["t_max"]
    x_in = torch.tensor(dataset.x_train[0])# [t0, t1, t2, t3]
    x_in = torch.reshape(x_in, (1, window_size)) #(batch_size, window_size)
    notes_pred = [x_in[0][0], x_in[0][1], x_in[0][2], x_in[0][3]]
    hidden_sz = hyperparams["hidden_sz"]
    vocab_sz =  hyperparams["vocab_sz"]
    all_sz = window_size + hidden_sz + hidden_sz + vocab_sz

    for i in range(t_max):
        if model_name == "mlp":
            x_in = x_in.float()
            x_out = model(x_in)
        elif model_name == "lstm":
            x_in = x_in.long()
            x_out = model(x_in)
        elif model_name == "feedback":
            x_in = x_in.float()
            if i == 0:
                # x_memory initialized as 0's
                x_memory = torch.zeros(x_in.size()[0], hyperparams["hidden_sz"])
                x_memory, x_out = model(x_in, x_memory)
                # detach memory from computation graph 
                x_memory = x_memory.detach()
            else:
                x_memory, x_out = model(x_in, x_memory)
                # detach memory from computation graph 
                x_memory = x_memory.detach()
        elif model_name == "eunet":
            x_in = x_in.float()
            if i == 0:
                # x_all initialized as 0's
                x_all = torch.zeros(x_in.size()[0], all_sz)
                x0, x1, x2, x_out  = model(x_in, x_all)
                # detach from computation graph 
                x_all = torch.cat((x0, x1, x2, x_out), -1).detach()
            else:
                x0, x1, x2, x_out  = model(x_in, x_all)
                # detach from computation graph 
                x_all = torch.cat((x0, x1, x2, x_out), -1).detach()
                
        y = torch.argmax(torch.flatten(x_out), -1)
        x_in  = torch.tensor([[notes_pred[-3],
                        notes_pred[-2],
                        notes_pred[-1],
                        y]])
                        
        notes_pred.append(y)
    
    notes_pred = [idx2note[n.item()] for n in notes_pred]
    correct = (notes_pred == song)
    acc = correct.sum()-4 / correct.size-4
    print("accuracy: %f " %(acc) + "%")
    play_notes(my_piano, np.array(notes_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--piano', default='data/my_piano.npy', help='Filepath to piano')
    parser.add_argument('-m', '--model', default='placeholder', help='Either mlp, lstm, feedback, eunet, or all')
    parser.add_argument("-t", "--train", action="store_true",
                        help="Run training loop")
    args = parser.parse_args()

    my_piano = np.load(args.piano, allow_pickle = True)

    # === music data
    song = np.array([67, 72, 74, 76, 76,
                    76, 74, 76, 72, 72, 
                    72, 74, 76, 77, 81, 
                    81, 79, 77, 76, 72,
                    74, 76, 77, 81, 81,
                    79, 77, 76, 72, 72,
                    74, 76, 77, 74, 74,
                    76, 72])

    play_notes(my_piano, song)

    
    dataset = SongDataset(song)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)

    idx2note = dataset.idx2note
    note2idx = dataset.note2idx
    hyperparams = {
        "num_epochs": 500,
        "batch_size": 1,
        "vocab_sz": len(dataset.note2idx),
        "embed_sz": 16,
        "rnn_sz": 16,
        "hidden_sz": 16,
        "t_max": len(dataset),
        "lr": 0.001
        }

    if args.model == "mlp":
        model = MLP(hyperparams)

    elif args.model == "lstm":
        model = LSTM(hyperparams)
    
    elif args.model == "feedback":
        model = FeedbackModel(hyperparams)
    
    elif args.model == "eunet":
        model = EuNet(hyperparams)
    
    elif args.model == "all":
        mlp_model = MLP(hyperparams)
        train(mlp_model, data_loader, "mlp")
        test(mlp_model, "mlp")

        feedback_model = FeedbackModel(hyperparams)
        train(feedback_model, data_loader, "feedback")
        test(feedback_model, "feedback")

        eunet = EuNet(hyperparams)
        train(eunet, data_loader, "eunet")
        test(eunet, "eunet")

        lstm_model = LSTM(hyperparams)
        train(lstm_model, data_loader, "lstm")
        test(lstm_model, "lstm")

    if args.train:
        train(model, data_loader, args.model)
        test(model, args.model)










