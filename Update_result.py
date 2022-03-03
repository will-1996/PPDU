import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FuncFormatter
# import pandas as pd
import copy
from matplotlib.pyplot import MultipleLocator


def Facebook_GccError():
    rabv = [0.7993, 0.709, 0.6212, 0.3296, 0.1205, 0.0531, 0.034, 0.0281]
    dgg = [0.2856, 0.2846, 0.2852, 0.2848, 0.284, 0.2849, 0.2854, 0.2842]
    ppdu = [0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(0, 1.0)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('GCC Error', fontsize=17)

    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_GccError()


def Facebook_ccError():
    rabv = [0.6864, 0.6489, 0.5637, 0.423, 0.3695, 0.2577, 0.1731, 0.1382]
    dgg = [0.2765, 0.284, 0.2827, 0.2786, 0.2802, 0.2775, 0.2788, 0.281]
    ppdu = [0.6329, 0.4984, 0.3269, 0.1779, 0.0855, 0.0371, 0.015, 0.0046]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(0, 1.0)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('CC Error', fontsize=17)

    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_ccError()

def Facebook_Mod():
    # rabv = np.array([0.6749, 0.7961, 0.7936, 0.8344, 0.835, 0.8338, 0.8349, 0.8349])
    # rabv = np.array([0.3653, 0.6724, 0.7626, 0.7553, 0.8251, 0.8309, 0.8333, 0.8339])
    rabv = np.array([0.1143, 0.2185, 0.4022, 0.5985, 0.7228, 0.7932, 0.8234, 0.8332])
    dgg = np.array([0.6514, 0.6511, 0.6505, 0.6509, 0.6512, 0.6511, 0.6507, 0.6513])
    ppdu = np.array([0.478, 0.63, 0.7406, 0.7967, 0.8199, 0.8282, 0.8326, 0.8342])
    ind = np.arange(8) + 1

    ground_true = 0.835
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, np.abs((ground_true - rabv) / ground_true), color='green', alpha=1, linestyle='--', linewidth=2, marker='x',
             markersize='7',
             label='SO-RNL')
    plt.plot(ind, (ground_true - dgg) / ground_true, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, np.abs((ground_true - ppdu) / ground_true), color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('Modularity Error', fontsize=17)

    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_Mod()

def Facebook_edge_add_Mod():
    # rabv = np.array([0.6749, 0.7961, 0.7936, 0.8344, 0.835, 0.8338, 0.8349, 0.8349])
    # rabv = np.array([0.3653, 0.6724, 0.7626, 0.7553, 0.8251, 0.8309, 0.8333, 0.8339])
    rabv = np.array([0.3572, 0.52, 0.6776, 0.7688, 0.809, 0.8249, 0.8314, 0.8334])
    dgg = np.array([0.7316, 0.7327, 0.7314, 0.7319, 0.7333, 0.7318, 0.7317, 0.7315])
    ppdu = np.array([0.79, 0.81, 0.83, 0.8343, 0.8334, 0.8339, 0.8344, 0.8346])
    ind = np.arange(8) + 1

    ground_true = 0.835
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, np.abs((ground_true - rabv) / ground_true), color='green', alpha=1, linestyle='--', linewidth=2, marker='x',
             markersize='7',
             label='SO-RNL')
    plt.plot(ind, (ground_true - dgg) / ground_true, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, np.abs((ground_true - ppdu) / ground_true), color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('Modularity Error', fontsize=17)

    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_edge_add_Mod()

def Facebook_edge_del_Mod():
    # rabv = np.array([0.6749, 0.7961, 0.7936, 0.8344, 0.835, 0.8338, 0.8349, 0.8349])
    # rabv = np.array([0.3653, 0.6724, 0.7626, 0.7553, 0.8251, 0.8309, 0.8333, 0.8339])
    rabv = np.array([0.850, 0.843, 0.837, 0.835, 0.834, 0.8274, 0.827, 0.827])
    dgg = np.array([0.863, 0.861, 0.86, 0.859, 0.859, 0.858, 0.858, 0.857])
    ppdu = np.array([0.848, 0.846, 0.837, 0.837, 0.832, 0.828, 0.828, 0.827])
    ind = np.arange(8) + 1

    ground_true = 0.8277
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, np.abs((ground_true - rabv) / ground_true), color='green', alpha=1, linestyle='--', linewidth=2, marker='x',
             markersize='7',
             label='SO-RNL')
    plt.plot(ind, np.abs((ground_true - dgg) / ground_true), color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, np.abs((ground_true - ppdu) / ground_true), color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.04, 0.1)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('Modularity Error', fontsize=17)

    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Facebook_edge_del_Mod()

def faceBook_ARMI():
    # ari_lf = np.array([0.1961, 0.2283, 0.269,  0.2787, 0.6447, 0.6974, 0.7254, 0.7449])+0.2
    # ami_lf = np.array([0.488, 0.5216, 0.5573, 0.6332, 0.6907, 0.7108, 0.728, 0.7371])+0.2
    # ari_ldpg = np.array([0.7105, 0.746, 0.7269, 0.7519, 0.7482, 0.7472, 0.7125, 0.7436])+0.2
    # ami_ldpg = np.array([0.729,  0.7232, 0.7365, 0.7502, 0.7524, 0.751, 0.7494, 0.7492])+0.2
    # ari_wdt = np.array([0.6021, 0.8252, 0.897, 0.9468, 0.9573, 0.963, 0.9897, 0.9974])+0.2
    # ami_wdt = np.array([0.7316, 0.8481, 0.9018, 0.9319, 0.9456, 0.9686, 0.9875, 0.9968])+0.2

    ari_lf = np.array([0.1961, 0.2283, 0.269, 0.2787, 0.6447, 0.6974, 0.7254, 0.7449])
    ami_lf = np.array([0.488, 0.5216, 0.5573, 0.6332, 0.6907, 0.7108, 0.728, 0.7371])
    ari_ldpg = np.array([0.7105, 0.746, 0.7269, 0.7519, 0.7482, 0.7472, 0.7125, 0.7436])
    ami_ldpg = np.array([0.729, 0.7232, 0.7365, 0.7502, 0.7524, 0.751, 0.7494, 0.7492])
    ari_wdt = np.array([0.6021, 0.8252, 0.897, 0.9468, 0.9573, 0.963, 0.9897, 0.9974])
    ami_wdt = np.array([0.7316, 0.8481, 0.9018, 0.9319, 0.9456, 0.9686, 0.9875, 0.9968])

    width = 0.8 / 3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.bar(np.arange( len(ari_lf)) + 1 - width, ari_ldpg, width=0.8 / 3, color='#4DBBD5CC',
                  label='ARI(DDGU)')

    lns2 = ax.bar(np.arange(len(ari_lf)) + 1 + 2 * width - width, ari_wdt, width=0.8 / 3, color='#F39B7FCC', label='ARI(PPDU)')

    lns3 = ax.bar(np.arange(len(ari_lf)) + 1 + width - width, ari_lf, width=0.8 / 3, color='#00A087CC',
                  label='ARI(SO-RNL)')

    ax2 = ax.twinx()
    lns4 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_ldpg, color='blue', label='AMI(DDGU)', linewidth=2,
                    marker='s', markersize='7')

    lns5 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_wdt, color='red', label='AMI(PPDU)', linewidth=2,
                    marker='o', markersize='7')

    lns6 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_lf, color='green', label='AMI(SO-RNL)', linewidth=2,
                    marker='^', markersize='7')

    fig.legend(loc=2, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
    # fig.legend(loc=(0.65, 0.11))
    # ax.grid()
    ax.set_xlabel('Privacy Budget', fontsize=17)
    ax.set_ylabel('ARI', fontsize=17)
    ax2.set_ylabel('AMI', fontsize=17)

    my_y_ticks = np.arange(-0.2, 1.2, 0.2)
    y_major_locator = MultipleLocator(0.2)
    # ax.yticks(my_y_ticks)
    # ax.set_ylim(-0.2, 1.2, ('100', '200', '300', '400', '500'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])

    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax2.set_yticklabels([ '0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    # ax2.set_ylim(-0.2, 1.2)
    plt.show()

# faceBook_ARMI()


def faceBook_edge_add_ARMI():
    # ari_lf = np.array([0.1961, 0.2283, 0.269,  0.2787, 0.6447, 0.6974, 0.7254, 0.7449])+0.2
    # ami_lf = np.array([0.488, 0.5216, 0.5573, 0.6332, 0.6907, 0.7108, 0.728, 0.7371])+0.2
    # ari_ldpg = np.array([0.7105, 0.746, 0.7269, 0.7519, 0.7482, 0.7472, 0.7125, 0.7436])+0.2
    # ami_ldpg = np.array([0.729,  0.7232, 0.7365, 0.7502, 0.7524, 0.751, 0.7494, 0.7492])+0.2
    # ari_wdt = np.array([0.6021, 0.8252, 0.897, 0.9468, 0.9573, 0.963, 0.9897, 0.9974])+0.2
    # ami_wdt = np.array([0.7316, 0.8481, 0.9018, 0.9319, 0.9456, 0.9686, 0.9875, 0.9968])+0.2

    ari_lf = np.array([0.5342, 0.742, 0.8905, 0.9721, 0.9764, 0.9836, 0.9846, 0.986])
    ami_lf = np.array([0.7571, 0.8047, 0.9019, 0.9682, 0.9755, 0.9795, 0.9816, 0.9842])
    ari_ldpg = np.array([0.8829, 0.8894, 0.8916, 0.9264, 0.9343, 0.936, 0.937, 0.943])
    ami_ldpg = np.array([0.8923, 0.9023, 0.9046, 0.926, 0.9284, 0.9292, 0.9313, 0.934])
    ari_wdt = np.array([0.9553, 0.9597, 0.9782,0.9793, 0.9851, 0.9856, 0.9877, 0.9887])
    ami_wdt = np.array([0.9614, 0.9674, 0.9688, 0.9782, 0.9809, 0.9856, 0.9858, 0.9884])

    width = 0.8 / 3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.bar(np.arange( len(ari_lf)) + 1 - width, ari_ldpg, width=0.8 / 3, color='#4DBBD5CC',
                  label='ARI(DDGU)')

    lns2 = ax.bar(np.arange(len(ari_lf)) + 1 + 2 * width - width, ari_wdt, width=0.8 / 3, color='#F39B7FCC', label='ARI(PPDU)')

    lns3 = ax.bar(np.arange(len(ari_lf)) + 1 + width - width, ari_lf, width=0.8 / 3, color='#00A087CC',
                  label='ARI(SO-RNL)')

    ax2 = ax.twinx()
    lns4 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_ldpg, color='blue', label='AMI(DDGU)', linewidth=2,
                    marker='s', markersize='7')

    lns5 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_wdt, color='red', label='AMI(PPDU)', linewidth=2,
                    marker='o', markersize='7')

    lns6 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_lf, color='green', label='AMI(SO-RNL)', linewidth=2,
                    marker='^', markersize='7')

    fig.legend(loc=(0.67, 0.11))
    # fig.legend(loc=(0.65, 0.11))
    # ax.grid()
    ax.set_xlabel('Privacy Budget', fontsize=17)
    ax.set_ylabel('ARI', fontsize=17)
    ax2.set_ylabel('AMI', fontsize=17)

    my_y_ticks = np.arange(-0.2, 1.2, 0.2)
    y_major_locator = MultipleLocator(0.2)
    # ax.yticks(my_y_ticks)
    # ax.set_ylim(-0.2, 1.2, ('100', '200', '300', '400', '500'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])

    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax2.set_yticklabels([ '0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    # ax2.set_ylim(-0.2, 1.2)
    plt.show()


# faceBook_edge_add_ARMI()