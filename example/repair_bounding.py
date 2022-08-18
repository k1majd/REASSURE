import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch, torchvision, time, math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from REASSURE.ExperimentModels import MLP
from REASSURE.Repair import REASSURERepair
from example.nnet_reader import HCAS_Model
from example.h5_reader import My_H5Dataset
from REASSURE.ExperimentTools import constraints_from_labels
import tensorflow as tf
import pickle


def success_rate(model, buggy_inputs, right_label, is_print=0):
    with torch.no_grad():
        pred = model(buggy_inputs)
        correct = (pred.argmax(1) == right_label).type(torch.float).sum().item()
    if is_print == 1:
        print(
            "Original accuracy on buggy_inputs: {} %".format(
                100 * correct / len(buggy_inputs)
            )
        )
    elif is_print == 2:
        print("Success repair rate: {} %".format(100 * correct / len(buggy_inputs)))
    return correct / len(buggy_inputs)


def original_data_loader():
    load_str = "_5_31_2022_16_35_50"
    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/global_constraint/data"
    ):
        os.makedirs(
            os.path.dirname(os.path.realpath(__file__))
            + "/repair_ex/global_constraint/data"
        )
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/repair_ex/global_constraint/data/repair_dataset{load_str}.pickle",
        "rb",
    ) as data:
        dataset = pickle.load(data)
    return dataset[0], dataset[1]
    # return dataset[0], dataset[1], dataset[2], dataset[3]


class neural_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=32, out_features=32)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=32, out_features=output_size)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)

        return x


def get_params(model_torch, weights):
    for m_perb, d in zip(model_torch.parameters(), weights):
        m_perb.data = torch.tensor(d).T
    return model_torch


def contstraint_bounding(num_samp):
    A = []
    b = []
    for i in range(num_samp):
        A.append(np.array([[1], [-1]]))
        b.append(np.array([10, 30]))
    return A, b


def Repair_HCAS(repair_num, n):

    # load the original model
    model_orig = tf.keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/global_constraint/model_orig/model_orig"
    )
    weights = model_orig.get_weights()
    # x_train, y_train, x_test, y_test = original_data_loader()
    x_train, y_train = original_data_loader()
    inp_max = []
    inp_min = []
    # for i in range(x_train.shape[1]):
    #     inp_max.append(np.maximum(x_train[:, i].max(), x_test[:, i].max()))
    #     inp_min.append(np.minimum(x_train[:, i].min(), x_test[:, i].min()))
    for i in range(x_train.shape[1]):
        inp_max.append(x_train[:, i].max())
        inp_min.append(x_train[:, i].min())
    inp_bound = [np.array(inp_max), -np.array(inp_min)]
    adv_idx = np.where(y_train > 10)[0]
    x_train = x_train[adv_idx]
    y_train = y_train[adv_idx]
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    # x_test = torch.tensor(x_test).float()
    # y_test = torch.tensor(y_test).float()
    model_torch = MLP(size=[40, 32, 32, 32, 1])
    model_torch = get_params(model_torch, weights)
    input_dim = 40
    input_boundary = [
        np.block([[np.eye(input_dim)], [-np.eye(input_dim)]]),
        np.block(inp_bound),
    ]
    # target_model = HCAS_Model(
    #     "example/TrainedNetworks/HCAS_rect_v6_pra1_tau05_25HU_3000.nnet"
    # )
    # buggy_inputs = torch.load("example/cex_pra1_tau05.pt")
    # buggy_inputs, right_labels = (
    #     buggy_inputs[:repair_num],
    #     torch.ones([repair_num], dtype=torch.long) * 4,
    # )
    output_constraints = contstraint_bounding(x_train[:repair_num].shape[0])
    # success_rate(model_torch, buggy_inputs, right_labels, is_print=1)
    start = time.time()
    REASSURE = REASSURERepair(model_torch, input_boundary, n)
    repaired_model = REASSURE.point_wise_repair(
        x_train, output_constraints=output_constraints, core_num=1
    )
    cost_time = time.time() - start
    print("Time:", cost_time)
    # success_rate(repaired_model, buggy_inputs, right_labels, is_print=2)


if __name__ == "__main__":
    for num in [50]:
        print("-" * 50, "number:", num, "-" * 50)
        Repair_HCAS(num, 1)
