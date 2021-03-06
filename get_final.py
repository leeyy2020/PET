import os
import sys
import numpy as np

from shutil import copyfile

if __name__=="__main__":
    result_file = sys.argv[1]
    output_dir = sys.argv[2]
    task_name = sys.argv[3]
    index = sys.argv[4]
    # submit_result_dir = sys.argv[5]

    # if not os.path.exists(submit_result_dir):
    #     os.mkdir(submit_result_dir)

    # dev_accs = []
    test_accs = []
    strategys = []
    epochs = []

    dev_num = 0
    #test_num = 0

    with open(result_file) as f:
        for line in f:
            # strategy_name, epoch, dev_acc, dev_num = line.strip().split("\t")
            strategy_name, epoch, test_acc, test_num = line.strip().split("\t")
            strategys.append(strategy_name)
            # dev_accs.append(float(dev_acc))
            test_accs.append(float(test_acc))
            epochs.append(epoch)

    # print("dev_accs:{}".format(dev_accs))
    print("test_accs:{}".format(test_accs))

    max_index = np.argmax(test_accs)
    # max_dev_acc = dev_accs[max_index]
    max_test_acc = test_accs[max_index]

    strategy = strategys[max_index]
    epoch = epochs[max_index]

    index_name = index if index != "few_all" else "all"

    if task_name not in ["eprstmt", "csldcp", "bustm"]:
        best_predict_file = "index" + index + "_" + epoch  + "epoch_" + task_name + "f_predict.json"
        std_name = task_name + "f_predict_" + index_name + ".json"
    else:
        best_predict_file = "index" + index + "_" + epoch  + "epoch_" + task_name + "_predict.json"
        std_name = task_name + "_predict_" + index_name + ".json"

    predict_file = os.path.join(output_dir, strategy, task_name, best_predict_file)
    print("best_result_file:{}".format(predict_file))

    # submit_file = os.path.join(submit_result_dir, std_name)
    # print("std_result_file:{}".format(submit_file))

    print("{}\t{}\t{}\t{}".format(strategy, epoch, max_test_acc, test_num))
    #暂时不需要提交所以注释
    # copyfile(predict_file, submit_file)
