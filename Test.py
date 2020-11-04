from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger
from Q__Learning import Q_learning
from Inma import Inma
import csv

for index in range(0,1):
    df = pd.read_csv("data/thaydoisonode.csv")
    node_pos = list(literal_eval(df.node_pos[index]))
    list_node = []
    random.seed(0)
    for i in range(len(node_pos)):

        location = node_pos[i]
        com_ran = df.commRange[index]
        energy = df.energy[index]
        energy_max = df.energy[index]
        prob = df.freq[index]
        node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                energy_thresh=0.4 * energy, prob=prob)
        list_node.append(node)
    file_name_target = "./log/Q_learning2nd/thaydoisonode/result_0_intervals/gamma0.4/regression_target_data" + str(index) + ".csv"
    information_log_t = open(file_name_target, "a+")
    writer_t = csv.DictWriter(information_log_t, fieldnames=["delta"])
    writer_t.writeheader()
    file_name_data = "./log/Q_learning2nd/thaydoisonode/result_0_intervals/gamma0.4/regression_data" + str(index) + ".csv"
    information_log_d = open(file_name_data, "a+")
    writer_d = csv.DictWriter(information_log_d, fieldnames=["E_ele", "M_ele"])
    writer_d.writeheader()
    file_name_w = "./log/Q_learning2nd/thaydoisonode/result_0_intervals/gamma0.4/regression_weights" + str(index) + ".csv"
    information_log_w = open(file_name_w, "a+")
    writer_w = csv.DictWriter(information_log_w, fieldnames=["Weights"])
    writer_w.writeheader()
    mc = MobileCharger(index=index, writer_t=writer_t, writer_d=writer_d, information_log_t=information_log_t, information_log_d=information_log_d, energy=df.E_mc[index], capacity=df.E_max[index], e_move=df.e_move[index],
                   e_self_charge=df.e_mc[index], velocity=df.velocity[index])
    target = [int(item) for item in df.target[index].split(',')]
    net = Network(index=index, list_node=list_node,  mc=mc, target=target)
    print(len(net.node), len(net.target), max(net.target))
    q_learning = Q_learning(writer_w=writer_w, information_log_w=information_log_w, network=net, index=index)
    inma = Inma()
    net.simulate(optimizer=q_learning, max_time=200000, file_name="./log/Q_learning2nd/thaydoisonode/inma_information" + str(index)+".csv")
