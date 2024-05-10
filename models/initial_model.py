from models.GCN import *
from models.LMF import *
from models.model import *

def init_model(num_views, num_classes, input_dims, latent_dims, num_layers, dropout, k_order, rank, output_dim, model_type):
    models = {}
    for i in range(num_views):
        gcn_key = f"G{i+1}"
        classifier_key = f"C{i+1}"
        models[gcn_key] = GCN(input_dims[i], latent_dims, k_order, num_layers, dropout, model_type)
        models[classifier_key] = Classifier(latent_dims[-1], num_classes)

    if num_views >= 2:
        models["F"] = LMF(latent_dims[-1], output_dim, rank)

    return models

def init_optim(num_views, models, learning_rate_gcn, learning_rate_lmf):
    optimizers = {}
    for i in range(num_views):
        gcn_key = f"G{i+1}"
        classifier_key = f"C{i+1}"
        combined_params = list(models[gcn_key].parameters()) + list(models[classifier_key].parameters())
        optimizers[classifier_key] = optim.Adam(combined_params, lr=learning_rate_gcn)

    if num_views >= 2:
        optimizers["F"] = optim.Adam(models["F"].parameters(), lr=learning_rate_lmf)

    return optimizers
