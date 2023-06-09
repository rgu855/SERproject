import torch
from torch import nn

class Baseline(nn.Module):
    def __init__(self, n_classes):
        super(Baseline, self).__init__()
        # Waveform encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 512, 10, stride=5),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, 2, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 768, 2, stride=2),
            nn.GELU(),
        )

        # Transformer encoding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=3072), num_layers=6)

        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.05)

        # Fully connected layer for classification
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x, mask = None):
        # Input is a waveform
        #print("Attention mask shape:", mask.shape)
        #print("Shape before conv:", x.shape)
        x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True)
        x = self.conv_layers(x)
        #print("Shape after conv:", x.shape)

        x = x.permute(2, 0, 1)
        #print("Shape before transformer:", x.shape)

        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        #print("Shape after transformer:", x.shape)
        x = x.mean(dim=0)
        #print("Shape before mlp:", x.shape)
        x = self.classifier(x)
        return x
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()


from transformers import HubertModel, HubertConfig

class TransferModel(nn.Module):
    def __init__(self, n_classes, pretrain = 'facebook/hubert-base-ls960', hidden_layers = None, layers_frozen = None):
        super(TransferModel, self).__init__()
        if hidden_layers is None:
            self.hubert_config = HubertConfig()
        else:
            self.hubert_config = HubertConfig(num_hidden_layers = hidden_layers)
        #print(self.hubert_config)
        self.hubert = HubertModel(self.hubert_config).from_pretrained(pretrain, config = self.hubert_config)
        #print(self.hubert.config)
        self.classifier = nn.Linear(768, n_classes)
        self.layers_frozen = layers_frozen
        self.hidden_layers = hidden_layers
        for name, param in self.named_parameters():
            if name.startswith("hubert.feature_extractor"):
                param.require_grad = False
                #print(f"{name} is frozen")
            elif name.startswith("hubert.encoder"):
                if layers_frozen is not None:
                    for layer_num in range(0, layers_frozen):
                        layer_name = "hubert.encoder.layers." + str(layer_num) +"."
                        if name.startswith(layer_name):
                            param.require_grad = False
                            #print(f"{name} is frozen")
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        output = self.hubert(x)[0]
        #print(output)
        output = output.mean(dim=1)
        output = self.classifier(output)
        return output
    
    def get_hidden_states(self, x):
        x = x.view(x.shape[0],-1)
        output = self.hubert(x)[0]
        #print(output)
        output = output.mean(dim=1)
        return output
    
class TransferredModel(nn.Module):
    def __init__(self, n_classes, transfer_model, freeze_all = False):
        super(TransferredModel, self).__init__()
        self.transfer_model = transfer_model
        self.classifier = nn.Linear(768, n_classes)
        if freeze_all == True:
            for name, param in self.named_parameters():
                if name.startswith("transfer_model.hubert.feature_extractor") or name.startswith("transfer_model.hubert.encoder"):
                    param.require_grad = False
                    #print(f"{name} is frozen")
        else:
            for name, param in self.named_parameters():
                if name.startswith("transfer_model.hubert.feature_extractor"):
                    param.require_grad = False
                    #print(f"{name} is frozen")
                elif name.startswith("transfer_model.hubert.encoder"):
                    if transfer_model.layers_frozen is not None:
                        for layer_num in range(0, transfer_model.layers_frozen):
                            layer_name = "transfer_model.hubert.encoder.layers." + str(layer_num) +"."
                            if name.startswith(layer_name):
                                param.require_grad = False
                                #print(f"{name} is frozen")
    def forward(self, x):
        output = self.transfer_model.get_hidden_states(x)
        output = self.classifier(output)
        return output
    
class ScratchModel(nn.Module):
    def __init__(self, n_classes):
        super(ScratchModel, self).__init__()
        self.hubert_config = HubertConfig(num_hidden_layers = 6)
        self.hubert = HubertModel(self.hubert_config)
        self.hubert.init_weights()
        self.classifier = nn.Linear(768, n_classes)

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        output = self.hubert(x)[0]
        #print(output)
        output = output.mean(dim=1)
        output = self.classifier(output)
        return output