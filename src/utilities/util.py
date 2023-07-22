import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import random
from collections import namedtuple

def calc_recalls(S):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images and columns are captions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def compute_pooldot_similarity_matrix(image_outputs, audio_outputs, nframes):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    S[i][j] is computed as the dot product between the meanpooled embeddings of
    the ith image output and jth audio output
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 4)
    n = image_outputs.size(0)
    imagePoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_image_outputs = imagePoolfunc(image_outputs).squeeze(3).squeeze(2)
    audioPoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_audio_outputs_list = []
    for idx in range(n):
        nF = max(1, nframes[idx])
        pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
    pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
    S = torch.mm(pooled_image_outputs, pooled_audio_outputs.t())
    return S

def one_imposter_index(i, N):
    imp_ind = random.randint(0, N - 2)
    if imp_ind == i:
        imp_ind = N - 1
    return imp_ind

def basic_get_imposter_indices(N):
    imposter_idc = []
    for i in range(N):
        # Select an imposter index for example i:
        imp_ind = one_imposter_index(i, N)
        imposter_idc.append(imp_ind)
    return imposter_idc

def semihardneg_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S as an autograd.Variable
    Output: The one-way triplet loss from rows of S to columns of S. Impostors are taken
    to be the most similar point to the anchor that is still less similar to the anchor
    than the positive example.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    loss = torch.autograd.Variable(torch.zeros(1).type(S.data.type()), requires_grad=True)
    # Imposter - ground truth
    Sdiff = S - torch.diag(S).view(-1, 1)
    eps = 1e-12
    # All examples less similar than ground truth
    mask = (Sdiff < -eps).type(torch.LongTensor)
    maskf = mask.type_as(S)
    # Mask out all examples >= gt with minimum similarity
    Sp = maskf * Sdiff + (1 - maskf) * torch.min(Sdiff).detach()
    # Find the index maximum similar of the remaining
    _, idc = Sp.max(dim=1)
    idc = idc.data.cpu()
    # Vector mask: 1 iff there exists an example < gt
    has_neg = (mask.sum(dim=1) > 0).data.type(torch.LongTensor)
    # Random imposter indices
    random_imp_ind = torch.LongTensor(basic_get_imposter_indices(N))
    # Use hardneg if there exists an example < gt, otherwise use random imposter
    imp_idc = has_neg * idc + (1 - has_neg) * random_imp_ind
    # This could probably be vectorized too, but I haven't.
    for i, imp in enumerate(imp_idc):
        local_loss = Sdiff[i, imp] + margin
        if (local_loss.data > 0).all():
            loss = loss + local_loss
    loss = loss / N
    return loss

def sampled_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S as an autograd.Variable
    Output: The one-way triplet loss from rows of S to columns of S. Imposters are
    randomly sampled from the columns of S.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    loss = torch.autograd.Variable(torch.zeros(1).type(S.data.type()), requires_grad=True)
    # Imposter - ground truth
    Sdiff = S - torch.diag(S).view(-1, 1)
    imp_ind = torch.LongTensor(basic_get_imposter_indices(N))
    # This could probably be vectorized too, but I haven't.
    for i, imp in enumerate(imp_ind):
        local_loss = Sdiff[i, imp] + margin
        if (local_loss.data > 0).all():
            loss = loss + local_loss
    loss = loss / N
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    print('now learning rate changed to {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        print('current learing rate is {:f}'.format(lr))
    lr = cur_lr  * 0.1
    print('now learning rate changed to {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def plot_single_result(data):
    class_performance_np = np.array(data)

    # Sort by performance
    sorted_indices = np.argsort(class_performance_np[:, 0])
    sorted_data = class_performance_np[sorted_indices]

    # Separate the class_ids and performances
    class_ids = sorted_data[:, 1]
    performances = sorted_data[:, 0]

    new_index = np.arange(len(class_ids))

    # Plot the data
    plt.figure(figsize=(10,6))
    plt.scatter(new_index, performances)
    plt.title('Class-wise Performance')
    plt.ylabel('mAP')
    plt.xlabel('Class Index')
    plt.show()
    
def plot_pair_data(data0, data1, label0, label1):
    plt.style.use('seaborn-whitegrid')
    class_performance = data0
    class_performance2 = data1
    # Convert the lists to numpy arrays
    class_performance_np = np.array(class_performance)
    class_performance_np2 = np.array(class_performance2)
    outperforms = np.sum(class_performance_np2  > class_performance_np)
    print(f"{outperforms} of {label1} outperformed {label0}")
    # Sort by performance for the first run
    sorted_indices = np.argsort(class_performance_np[:, 0])
    sorted_data = class_performance_np[sorted_indices]

    # Separate the class_ids and performances
    class_ids = sorted_data[:, 1]
    performances = sorted_data[:, 0]

    # Create new index according to sorted order
    new_index = np.arange(len(class_ids))

    # Create a mapping from class_id to new index
    class_id_to_new_index = {class_id: index for index, class_id in enumerate(class_ids)}

    # Get the corresponding new indices for the second run
    new_index2 = np.array([class_id_to_new_index[class_id] for class_id in class_performance_np2[:, 1]])
    performances2 = class_performance_np2[:, 0]

    # Fit a loess curve to the data
    loess = sm.nonparametric.lowess(performances2, new_index2, frac=0.3)

    # Plot the data and the loess curve
    plt.figure(figsize=(10,6))
    plt.scatter(new_index, performances, color='blue', alpha=0.2, label=label0)

    plt.scatter(new_index2, performances2, color='red', alpha=0.2, label=label1)
    plt.plot(loess[:, 0], loess[:, 1], color='red', linestyle='dotted', linewidth=3)
    plt.title('Class-wise Performance')
    plt.ylabel('mAP')
    plt.xlabel('Class Index')
    plt.legend()
    plt.savefig(f'performance_classwise_{label0}_{label1}.png', dpi=300)

    plt.show()
    return FileLink(f'performance_classwise_{label0}_{label1}.png')

def calculate_result(new_gas_eval):
    result = []
    batch_size =100
    x = new_gas_eval 
    # x = new_gas_eval
    # x = gas_eval_x1_origin
    for i in range(0, len(x), batch_size):
        with torch.no_grad():
            input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
            output = model.forward(input)
    #                 print(output)
            output = torch.sigmoid(output)
    #                 print(f'output shape: {output.shape}')
    #                 print(output)
            result.append([output.data.cpu().numpy()])
    print('total num of batches during testing', len(result))
    result = [numpy.concatenate(items) for items in zip(*result)]
    gas_eval_global_prob_multi = result[0]
    print('Performance on Google Audio Set:')
    print("   CLASS |    AP |   AUC |    d' ")
    FORMAT  = ' %00007s | %5.3f | %5.3f |%6.03f '
    SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
    print(SEP)
    classwise = []
    N_CLASSES = gas_eval_global_prob_multi[0].shape[-1]
    for i in range(N_CLASSES):
        a, b, c = gas_eval(gas_eval_global_prob_multi[:,i], gas_eval_y[:,i])     # AP, AUC, dprime
        classwise.append((a, b, c))
    map, mauc = numpy.array(classwise).mean(axis = 0)[:2]
    print(FORMAT % ('Average', map, mauc, dprime(mauc)))
    print(SEP)
    for i in range(N_CLASSES):
        print(FORMAT % ((str(i),) + classwise[i]))
    classwise_data = [[v[0], i] for i, v in enumerate(classwise)]
    return classwise_data, gas_eval_global_prob_multi

def calculate_result_torch(new_gas_eval):
    result = []
    batch_size = 100
    x = new_gas_eval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for i in range(0, len(x), batch_size):
        with torch.no_grad():
            input = torch.tensor(x[i : i + batch_size]).to(device)
            output = model.forward(input)
            output = torch.sigmoid(output)
            result.append(output.data)
    print('total num of batches during testing', len(result))
    result_tensor = torch.cat(result)
    gas_eval_global_prob = result_tensor.cpu().numpy()
    print('Performance on Google Audio Set:')
    print("   CLASS |    AP |   AUC |    d' ")
    FORMAT  = ' %00007s | %5.3f | %5.3f |%6.03f '
    SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
    print(SEP)
    classwise = []
    N_CLASSES = gas_eval_global_prob.shape[-1]
    for i in range(N_CLASSES):
        a, b, c = gas_eval(gas_eval_global_prob[:,i], gas_eval_y[:,i])     # AP, AUC, dprime
        classwise.append((a, b, c))
    map, mauc = np.array(classwise).mean(axis = 0)[:2]
    print(FORMAT % ('Average', map, mauc, dprime(mauc)))
    print(SEP)
    for i in range(N_CLASSES):
        print(FORMAT % ((str(i),) + classwise[i]))
    classwise_data = [[v[0], i] for i, v in enumerate(classwise)]
    return classwise_data, gas_eval_global_prob

def plot_binary_confusion(glob_prob):
    # convert to binary labels
    glob_prob_binary = np.where(glob_prob > 0.5, 1, 0)

    # calculate the multilabel confusion matrix
    confusion_matrices = multilabel_confusion_matrix(gas_eval_y, glob_prob_binary)

    classes_to_plot = [0, 1, 2, 3, 4]  # replace with your chosen classes

    for i in classes_to_plot:
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrices[i], annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Class {i}')
        plt.show()
        
def multiclass_confusion_matrix_numpy(y_true, y_pred):
    conf_matrix = np.zeros((y_true.shape[-1], y_true.shape[-1]))
    for actual, pred in zip(y_true, y_pred):
        act_indices =  np.nonzero(actual)[0]
        k = act_indices.shape[0]
        topk_indices = np.argsort(pred)[-k:]
        TP = np.intersect1d(act_indices,topk_indices)
        for hit_ind in TP:
            conf_matrix[hit_ind][hit_ind]+=1
        FN = np.setdiff1d(act_indices, topk_indices)
        FP = np.setdiff1d(topk_indices, act_indices)
        mistakes = [(fn, fp) for fn in FN for fp in FP]
        for act_i, pred_i in mistakes:
            conf_matrix[act_i][pred_i]+=1
    return conf_matrix

def visualize_confusion(confusion_matrix, start_ind=0, end_ind=527):
    # cm_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    cm_normalized_all = confusion_matrix.astype('float')
    cm_log_scale = np.log1p(cm_normalized_all)

    plt.figure(figsize=(15, 15))
    plt.imshow(cm_log_scale[start_ind:end_ind, start_ind:end_ind] , interpolation='nearest', cmap=plt.cm.Blues)
    # plt.imshow(cm_normalized[250:350, 250:350], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", dpi=300)

    plt.show()

    # Display a link to the saved file
    return FileLink('confusion_matrix.png')

def topk_classes_confuse(confusion, axis, k):
    column_sums = confusion.sum(axis=axis)
    top_k_indices = np.argsort(column_sums)[-k:]
    for ind in top_k_indices:
        print(ind, class_dic[str(ind)])
    plt.plot(column_sums)
    
def naive_confusion_matrix(y_true, y_pred):
#     y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    print(y_true.shape, y_pred.shape)
    # Convert the probability predictions into class predictions
    pred_classes = np.argmax(y_pred, axis=1)
    # sorted_indices = np.argsort(glob_prob0, axis=1)
    # pred_classes = sorted_indices[:, -2]

    true_classes = np.argmax(y_true, axis=1)
    # sorted_truth = np.argsort(gas_eval_y, axis=1)
    # true_classes = sorted_truth[:, -2]
    print(pred_classes.shape, true_classes.shape)
    all_classes = np.arange(527)

    # Compute the confusion matrix
    cm = confusion_matrix(true_classes, pred_classes, labels=all_classes)
    print(cm.shape)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_log = np.log1p(cm)

    plt.figure(figsize=(15, 15))
    plt.imshow(cm_log, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('1 hot confusion matrix')
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("1-hot-confusion_matrix.png", dpi=300)
    # Display a link to the saved file
    plt.show()
    return FileLink('1-hot-confusion_matrix.png'), cm

def plot_precision_recall_f1(y_true, y_pred):# convert to binary labels
    glob_prob_binary = np.where(y_pred > 0.5, 1, 0)

    # calculate precision, recall, F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, glob_prob_binary)

    # get sorted order by F1 score
    sorted_classes = np.argsort(f1)[::-1]  # reverse for descending order

    # plot precision, recall, F1 for each class in sorted order
    plt.figure(figsize=(15, 10))
    plt.plot(precision[sorted_classes], label='Precision')
    plt.plot(recall[sorted_classes], label='Recall')
    plt.plot(f1[sorted_classes], label='F1')
    plt.xlabel('Class (sorted by F1 score)')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score for Each Class')
    plt.legend()
    plt.show()
    plt.show()

PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])

RNNConfig = namedtuple(
  'RNNConfig',
  ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])
