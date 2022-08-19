import sys, os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch, torchvision, time, math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    load_str = "_6_9_2022_21_32_23"
    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/dynamic_constraint/data"
    ):
        os.makedirs(
            os.path.dirname(os.path.realpath(__file__))
            + "/repair_ex/dynamic_constraint/data"
        )
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/repair_ex/dynamic_constraint/data/repair_dataset{load_str}.pickle",
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


def contstraint_bounding(x, bound=1.5):
    num_samp = x.shape[0]
    A = []
    b = []
    for i in range(num_samp):
        A.append(np.array([[1], [-1]]))
        b.append(
            np.array(
                [bound + x[i, -1].detach().numpy(), bound - x[i, -1].detach().numpy()]
            )
        )
    return A, b


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(list(csv.reader(csv_file, delimiter=",")), dtype=np.float32)
    return data


def generateDataWindow(window_size):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/dynamic_constraint/data/GeoffFTF_1.csv"
    )
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/dynamic_constraint/data/GeoffFTF_2.csv"
    )
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/dynamic_constraint/data/GeoffFTF_3.csv"
    )
    n = 20363
    Dankle = np.subtract(Dtib[: n + 1, 1], Dfut[: n + 1, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    observations = np.concatenate(
        (
            observations,
            Dankle[:n].reshape(n, 1),
        ),
        axis=1,
    )
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18200
    # n_train = 500
    train_observation = np.array([]).reshape(0, 5 * window_size)
    test_observation = np.array([]).reshape(0, 5 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate((train_observation, temp_obs), axis=0)
    train_controls = controls[window_size : n_train + window_size].reshape(-1, 1)
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)
    return (
        train_observation,
        train_controls,
        test_observation,
        test_controls[:-1],
    )


def Repair_HCAS(repair_num, n):
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)

    # load the original model
    model_orig = tf.keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/dynamic_constraint/model_orig/model_orig"
    )
    weights = model_orig.get_weights()
    # x_train, y_train, x_test, y_test = original_data_loader()

    # data generation
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

    delta_u = model_orig.predict(x_train).flatten() - x_train[:, -1].flatten()
    adv_idx = np.where(np.abs(delta_u) > 1.9)[0]
    x_train = x_train[adv_idx]
    y_train = y_train[adv_idx]
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    # x_test = torch.tensor(x_test).float()
    # y_test = torch.tensor(y_test).float()
    model_torch = MLP(size=[50, 32, 32, 32, 1])
    model_torch = get_params(model_torch, weights)
    input_dim = 50
    input_boundary = [
        np.block([[np.eye(input_dim)], [-np.eye(input_dim)]]),
        np.block(inp_bound),
    ]

    # bound = 2
    # output_constraints = contstraint_bounding(x_train[:repair_num], bound)
    # x_train = x_train[:repair_num]
    # # success_rate(model_torch, buggy_inputs, right_labels, is_print=1)
    # start = time.time()
    # REASSURE = REASSURERepair(model_torch, input_boundary, n)
    # repaired_model = REASSURE.point_wise_repair(
    #     x_train, output_constraints=output_constraints, core_num=1
    # )
    # cost_time = time.time() - start
    x_test = torch.tensor(test_obs).float()
    y_test = torch.tensor(test_ctrls).float()
    # y_pred = repaired_model(x_test)
    # plt.plot(
    #     y_test.detach().numpy().flatten() - x_test[:, -1].detach().numpy().flatten(),
    #     label="true",
    # )
    # plt.plot(
    #     y_pred.detach().numpy().flatten() - x_test[:, -1].detach().numpy().flatten(),
    #     label="pred",
    # )
    # plt.legend()
    # plt.show()
    # print("Time:", cost_time)
    # pickle.dump(repaired_model, open("model_torch_dynamic_1_5.pkl", "wb"))
    model = pickle.load(open("model_torch_dynamic_2.pkl", "rb"))
    y_pred = model(x_test)
    x_test = x_test.detach().numpy()
    y_pred = y_pred.detach().numpy()
    delta_u = y_pred.flatten() - x_test[:, -1].flatten()
    pickle.dump([y_pred, delta_u], open("model_torch_2_data.pkl", "wb"))

    # model = pickle.load(open("model_torch_dynamic_2.pkl", "rb"))
    # y_pred = model(x_test)
    # x_test = x_test.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    # delta_u = y_pred.flatten() - x_test[:, -1].flatten()
    # pickle.dump([y_pred, delta_u], open("model_torch_2_data.pkl", "wb"))
    # success_rate(repaired_model, buggy_inputs, right_labels, is_print=2)


if __name__ == "__main__":
    for num in [25]:
        print("-" * 50, "number:", num, "-" * 50)
        Repair_HCAS(num, 1)
