#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import yaml

def load_config(path):
    data_loaded = None

    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded

def pretty_print_list(iterable):
    for e in iterable:
        print("  -", e)

def save_config(dic, folder):
    config = dict()
    parameters = dict()

    for i in dic:
        child = i["props"]["children"]

        name  = child[0]["props"]["children"]
        value = child[1]["props"]["value"]
        if name != "action_space":
            parameters[name] = value
        else:
            config[name] = value
    config["parameters"] = parameters

    with open(folder + "/agent_config.yaml", 'w') as f:
        yaml.dump(config, f)

    pretty_print_dic(config, 1)
        
def pretty_print_dic(dic, nb):
    print(" " * nb, "{")
    for e in list(dic.keys()):
        print(" " * nb, "-", e, ":")
        if type(dic[e]) == dict:
            pretty_print_dic(dic[e], nb + 1)
        elif type(dic[e]) == list:
            pretty_print_list_2(dic[e], nb + 1)
        else:
            print("  " * (nb + 1), dic[e])
    print(" " * nb, "}")

def pretty_print_list_2(iterable, nb):
    print(" " * nb, "[")
    for e in iterable:
        if type(e) == dict:
            pretty_print_dic(e, nb + 1)
        elif type(e) == list:
            pretty_print_list_2(e, nb + 1)
        else:
            print("  " * nb, "-", e)
    print(" " * nb, "]")


