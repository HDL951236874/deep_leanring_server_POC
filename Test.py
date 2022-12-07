import torch

import dataloader
import register
import world
from Procedure import Test
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/bd')
def app1():
    datasetpath = "./ml-100k"
    config = {'bpr_batch_size': 2048, 'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0, 'keep_prob': 0.6,
              'A_n_fold': 100, 'test_u_batch_size': 100, 'multicore': 0, 'lr': 0.001, 'decay': 0.0001, 'pretrain': 0,
              'A_split': False, 'bigdata': False}
    dataset = dataloader.Loader(config=config, path=datasetpath)

    Recmodel = register.MODELS["lgn"](config, dataset)
    Recmodel = Recmodel.to(world.device)
    # weight_file = utils.getFileName()
    weight_file = "/homework/Light-GCN-in-the-Movie-Recommender-System-main\\code\\checkpoints\\02122022_2216-lgn-ml-100k-3-64.pth.tar"
    Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

    Test(dataset=dataset, Recmodel=Recmodel, epoch=1, w=None)
    return {
        "msg": "success",
        "data": 1
    }


if __name__ == '__main__':
    app.run()
