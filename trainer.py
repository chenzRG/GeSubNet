import os
import tqdm
from torch_geometric.utils import negative_sampling
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def train_SELM(SELM_model, train_dataloader, test_dataloader, device, log_dir, epoch = 100):
    Classifier_loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(GRN_model.parameters(), lr=1e-4, amsgrad=False)
    loss_list = []
    val_loss_list = []
    ac_list = []
    for epoch in tqdm.tqdm(range(epoch)):
        running_loss = 0.0
        count=0
        for _, (inputs, labels) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            _, rec, cls = GRN_model(inputs)
            rec_loss = F.mse_loss(rec, inputs)
            cls_loss = Classifier_loss(cls,labels.squeeze())
            loss = rec_loss+cls_loss
            
            loss.backward()
            optimizer.step()
            count=count+1

            running_loss += loss.item()
        loss_loss=running_loss/count
        loss_list.append(loss_loss)
        print('epoch',epoch+1,':finished')
        print('train_loss:',loss_loss)
        with torch.no_grad():
            count=0
            running_loss=0.0
            pre=list()
            lab=list()
            for _, (inputs, labels) in enumerate(test_dataloader, 0):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                _, rec, cls = GRN_model(inputs)
                rec_loss = F.mse_loss(rec, inputs)
                cls_loss = Classifier_loss(cls,labels.squeeze())
                loss = rec_loss+cls_loss
                running_loss += loss.item()
                count+=1
                _, predicted = torch.max(F.softmax(cls).data, dim=1)
                predicted=predicted.to('cpu')
                labels=labels.to('cpu')
                predicted=predicted.tolist()
                labels=labels.tolist()
                pre.append(predicted)
                lab.append(labels)
            loss_loss=running_loss/count
            val_loss_list.append(loss_loss)
            pre=sum(pre,[])
            lab=sum(lab,[])
            print('val_loss:',loss_loss)
            cl = classification_report(lab, pre,output_dict=True)
            print(cl)
            ac_list.append(cl['accuracy'])

    # Create a 2x1 subplot
    figure, axs = plt.subplots(2, 1, figsize=(8, 5))

    # Plot the loss and validation loss in the first subplot (P1)
    axs[0].plot(loss_list[1:], label='Train Loss')
    axs[0].plot(val_loss_list[1:], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Train Loss and Validation Loss')

    # Plot the accuracy in the second subplot (P2)
    axs[1].plot(ac_list, label='Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].set_title('Accuracy')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    os.makedirs(os.path.join(log_dir), exist_ok=True)
    plt.savefig(os.path.join(log_dir, "AE_GRN.png"))
    plt.close('all')
    torch.save(GRN_model.state_dict(), os.path.join(log_dir, f'GRN.pt'))
    return ac_list[-1]

def train_link_predictor(GNN_model, train_data, val_data, A, log_dir, n_epochs=200):
    optimizer = torch.optim.Adam(params=GNN_model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_losses = []
    val_aucs = []

    for epoch in tqdm.tqdm(range(1, n_epochs + 1)):
        GNN_model.train()
        optimizer.zero_grad()
        z = GNN_model.encode(train_data.x, train_data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse'
        )

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = GNN_model.decode(z, edge_label_index, A).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(GNN_model, val_data, A)

        train_losses.append(loss.item())
        val_aucs.append(val_auc)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    os.makedirs(os.path.join(log_dir, "saved_models"), exist_ok=True)
    plt.savefig(os.path.join(log_dir, "link_predictor.png"))
    plt.close('all')
    torch.save(GNN_model.state_dict(), os.path.join(log_dir, "saved_models",'new_test_disc_1'))

@torch.no_grad()
def eval_link_predictor(GNN_model, data, A):
    GNN_model.eval()
    z = GNN_model.encode(data.x, data.edge_index)
    out = GNN_model.decode(z, data.edge_label_index, A).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def minimize_MSE(GNN_model, device, part, input_parts_dict, DGM_parts_dict, devided_graph, log_dir, epoch = 200):
    #GNN_model.load_state_dict(torch.load(os.path.join(data_dir,'new_test_disc_1')))
    optimizer = torch.optim.Adam(params=GNN_model.parameters(), lr=0.01)
    loss_list = []
    count = 0  
    
    input_data = torch.tensor(input_parts_dict[part], dtype=torch.float32).to(device)  
    DGM_z = torch.tensor(DGM_parts_dict[part], dtype=torch.float32, device=device)

    for epoch in tqdm.tqdm(range(epoch)):
        running_loss = 0.0
        GNN_model.train()
        optimizer.zero_grad()
        z = GNN_model.encode(devided_graph.to(device).x, devided_graph.to(device).edge_index)
        z = z.to(torch.float32)
        rec = torch.matmul(z, DGM_z)
        rec_loss = F.mse_loss(rec, input_data)
        loss = rec_loss
        loss.backward()
        optimizer.step()
        count = count + 1
        running_loss += loss.item()
        loss_loss = running_loss / count
        loss_list.append(loss_loss)
        if epoch % 10 == 0:
            print('epoch', epoch + 1, ': finished')
            print('train_loss:', loss_loss)
    torch.save(GNN_model.state_dict(), os.path.join(log_dir, "saved_models",'new_test_disc_1_step2'))