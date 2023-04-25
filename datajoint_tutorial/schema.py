import datajoint as dj
import torch.optim as optim
from datajoint_tutorial.torch_network import Net, get_dataloaders, train, test

dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'password'

schema = dj.schema('tutorial', locals())

@schema
class NumFeatures(dj.Manual):
    definition = """
    features_config_id  : tinyint # so-called primary key, must be unique
    ---
    num_features_1      : smallint
    num_features_2      : smallint
    """

@schema
class DropoutProb(dj.Lookup):
    definition = """
    dropout_config_id  : tinyint # so-called primary key, must be unique
    ---
    dropout_prob       : float
    """

@schema
class LearningRate(dj.Lookup):
    definition = """
    lr_config_id  : tinyint # so-called primary key, must be unique
    ---
    lr            : float
    """

@schema
class NumEpochs(dj.Lookup):
    definition = """
    epochs_config_id  : tinyint # so-called primary key, must be unique
    ---
    epochs            : int
    """

@schema
class Train(dj.Computed):
    definition = """
    -> NumFeatures
    -> DropoutProb
    -> LearningRate
    -> NumEpochs
    ---
    train_loss      : float
    """
    
    class Weights(dj.Part):
        definition = """  # weights of the trained model
        -> Train
        layer     : varchar(64)   # layer name
        ---
        weights  : longblob      # numpy array of model weigths
        """
        
    def make(self, key):
        train_loader, test_loader = get_dataloaders(batch_size=64)
        
        num_features_1, num_features_2 = (NumFeatures() & key).fetch1("num_features_1", "num_features_2")
        dropout_prob = (DropoutProb() & key).fetch1("dropout_prob")
        lr = (LearningRate() & key).fetch1("lr")
        num_epochs = (NumEpochs() & key).fetch1("epochs")
        
        model = Net(num_features_1=num_features_1, num_features_2=num_features_2, dropout_prob=dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(1, num_epochs + 1):
            loss = train(model, train_loader, optimizer, epoch)
            
        key["train_loss"] = float(loss.detach().numpy())
        self.insert1(key)
        del key["train_loss"]
        
        for k, v in model.state_dict().items():
            key["layer"] = k
            key["weights"] = v.numpy()
            self.Weights.insert1(key)
