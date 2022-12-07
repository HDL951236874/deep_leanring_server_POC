# -*- coding: utf-8 -*-
# @Author  : DAOLIN HAN
# @Time    : 2022/12/4 17:08
# @Function:
import torch

import Procedure
import dataloader
import register
import utils
import world

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/<int:user>', methods=['GET'])
def finetuning_process(user):
    datasetpath = "./ml-100k"
    config = {'bpr_batch_size': 2048, 'latent_dim_rec': 64, 'lightGCN_n_layers': 3, 'dropout': 0,
              'keep_prob': 0.6, 'A_n_fold': 100, 'test_u_batch_size': 100,
              'multicore': 0, 'lr': 0.001, 'decay': 0.0001, 'pretrain': 0, 'A_split': False,
              'bigdata': False}
    dataset = dataloader.Loader(config=config, path=datasetpath)
    bar = request.args.to_dict()

    Recmodel = register.MODELS["lgn"](config, dataset)
    Recmodel = Recmodel.to("cpu")
    weight_file = "code/checkpoints/02122022_2216-lgn-ml-100k-3-64.pth.tar"
    Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    bpr = utils.BPRLoss(Recmodel, config)
    batch_users = [int(user),bar]
    batch_users_gpu = torch.Tensor(batch_users).long()
    batch_users_gpu = batch_users_gpu.to("cpu")
    rating = Recmodel.getUsersRating(batch_users_gpu)
    _, rating_K = torch.topk(rating, k=5)
    rating_K = rating_K.cpu().numpy()

    return jsonify({'ans': str(rating_K)})


if __name__ == '__main__':
    app.run(host="0.0.0.0")
