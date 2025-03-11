import torch

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data1, label ,transform = None):
        self.transform = transform
        self.data1 = data1
        self.label = label
        self.datanum = len(data1)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        
        out_data1 = torch.tensor(self.data1[idx]).float() 
        out_label = torch.tensor(self.label[idx])
        if self.transform:
            out_data1 = self.transform(out_data1)

        return out_data1, out_label