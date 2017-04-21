import os
import fixed_point_training
def dump_to_txt_files(pt_acc_list, acc_list):
    with open("ptacc_cifar_quantize_han.txt", "w") as f:
        for item in pt_acc_list:
            f.write("%s\n"%item)
    with open("acc_cifar_quantize_han.txt", "w") as f:
        for item in acc_list:
            f.write("%s\n"%item)

acc_list = []
pt_acc_list = []
count = 0
retrain = 0
parent_dir = './'
base_model = 'base.pkl'
quantisation_bits = [4,8,16,32,64]
pcov = [0,0]
for q_width in quantisation_bits:
    # measure acc
    param = [
        ('-t', 0),
        ('-q_bits',q_width),
        ('-pretrain',1),
        ('-parent_dir', parent_dir),
        ('-base_model', base_model)
        ]
    pre_train_acc = fixed_point_training.main(param)
    param = [
        ('-t', 1),
        ('-q_bits',q_width),
        ('-pretrain',1),
        ('-parent_dir', parent_dir),
        ('-base_model', base_model)
        ]
    _ = fixed_point_training.main(param)

    param = [
        ('-t', 0),
        ('-q_bits',q_width),
        ('-pretrain',0),
        ('-parent_dir', parent_dir),
        ('-base_model', base_model)
        ]
    train_acc = fixed_point_training.main(param)
    pt_acc_list.append(pre_train_acc)
    acc_list.append(train_acc)
    print(pt_acc_list)
    print(acc_list)
    dump_to_txt_files(pt_acc_list, acc_list)
    count = count + 1
print('accuracy summary: {}'.format(pt_acc_list))
print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
