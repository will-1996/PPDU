import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FuncFormatter
# import pandas as pd
import copy
from matplotlib.pyplot import MultipleLocator


def Facebook_GccError():
    num100 = [0.0387, 0.0174, 0.0078, 0.0022, 0.001, 0.0004, 0.0001, 0.0]
    num200 = [0.2804, 0.1646, 0.0883, 0.0408, 0.0161, 0.0063, 0.0026, 0.0016]
    num500 = [0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009]
    num1000 = [0.7981, 0.6699, 0.3848, 0.164, 0.0619, 0.0232, 0.0078, 0.0028]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, num100, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='100 new nodes')
    plt.plot(ind, num200, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='200 new nodes')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, num500, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='500 new nodes')
    plt.plot(ind, num1000, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
             label='1000 new nodes')
    plt.ylim(0, 1.0)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('GCC Error', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_GccError()


def Facebook_GccError_delta():
    delta1 = [0.5032, 0.2682, 0.1176, 0.0492, 0.0181, 0.0065, 0.0028, 0.001]
    delta2 = [0.6621, 0.4214, 0.2137, 0.0896, 0.0346, 0.0133, 0.0053, 0.0017]
    delta3 = [0.7338, 0.5282, 0.2919, 0.1282, 0.0502, 0.019, 0.0071, 0.0025]
    delta4 = [0.7647, 0.5963, 0.3571, 0.1656, 0.0682, 0.0252, 0.0104, 0.0034]
    delta20 = [0.8322, 0.7638, 0.7188, 0.5, 0.2578, 0.1075, 0.0433, 0.0156]

    rabv =   [0.7443, 0.6761, 0.1665, 0.0541, 0.0134, 0.0006, 0.0021, 0]
    num200 = [0.2804, 0.1646, 0.0883, 0.0408, 0.0161, 0.0063, 0.0026, 0.0016]
    num500 = [0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009]
    num1000 = [0.7981, 0.6699, 0.3848, 0.164, 0.0619, 0.0232, 0.0078, 0.0028]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, delta1, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='$\eta = 1$ (PPDU)')
    plt.plot(ind, delta2, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='$\eta = 2$ (PPDU)')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, delta4, color='orange', alpha=1, linewidth=2, marker='o', markersize='7', label='$\eta = 4$ (PPDU)')
    plt.plot(ind, delta20, color='red', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
             label='Naive RNL')
    plt.plot(ind, rabv, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    # plt.plot(ind, delta20, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
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

    plt.legend(loc='upper right', fontsize=15)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_GccError_delta()



def Enron_GccError_delta():
    delta1 = [0.4363, 0.2087, 0.0933, 0.037, 0.0133, 0.0047, 0.0022, 0.0007]
    delta2 = [0.6045, 0.3519, 0.1669, 0.0676, 0.0286, 0.0104, 0.0034, 0.0018]
    delta3 = [0.6988, 0.457, 0.2329, 0.1011, 0.0391, 0.0156, 0.0049, 0.0021]
    delta4 = [0.7528, 0.5352, 0.2925, 0.1291, 0.0529, 0.0195, 0.0072, 0.0026]
    delta20 = [0.844, 0.8292, 0.6959, 0.449, 0.22, 0.0924, 0.0352, 0.0133]

    rabv =  [0.9846, 0.9654, 0.8988, 0.424, 0.0494, 0.0061, 0.001, 0]

    rnl = [0.99, 0.985, 0.965, 0.85, 0.46, 0.096,  0.0352, 0.0133]
    num200 = [0.2804, 0.1646, 0.0883, 0.0408, 0.0161, 0.0063, 0.0026, 0.0016]
    num500 = [0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009]
    num1000 = [0.7981, 0.6699, 0.3848, 0.164, 0.0619, 0.0232, 0.0078, 0.0028]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, delta1, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='$\eta = 1$ (PPDU)')
    plt.plot(ind, delta2, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='$\eta = 2$ (PPDU)')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, delta4, color='orange', alpha=1, linewidth=2, marker='o', markersize='7', label='$\eta = 4$ (PPDU)')
    plt.plot(ind, rnl, color='red', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
             label='Naive RNL')
    plt.plot(ind, rabv, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    # plt.plot(ind, delta20, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.ylim(0, 1)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('GCC Error', fontsize=17)

    plt.legend(loc='upper right', fontsize=15)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Enron_GccError_delta()


def Facebook_Mod_EdgeInser_times():     # RNL的颜色统一为蓝色，但是
    delta1 = np.array([0.3572, 0.52, 0.6776, 0.7688, 0.809, 0.8249, 0.8314, 0.8334])
    delta2 = np.array([0.2913, 0.352, 0.4313, 0.5213, 0.6065, 0.6775, 0.731, 0.7668])
    # delta3 = [0.7338, 0.5282, 0.2919, 0.1282, 0.0502, 0.019, 0.0071, 0.0025]
    delta4 = np.array([0.2649, 0.29, 0.3193, 0.3538, 0.392, 0.4332, 0.4783, 0.5236])
    delta8 = np.array([0.2538, 0.2627, 0.2795, 0.2901, 0.3066, 0.3185, 0.3413, 0.3526])

    # ppdu = np.array([0.79, 0.81, 0.83, 0.8343, 0.8334, 0.8339, 0.8344, 0.8346])
    ppdu4 = np.array([0.78, 0.80, 0.825, 0.83, 0.8334, 0.8339, 0.8344, 0.8346])

    # rabv =   [0.7443, 0.6761, 0.1665, 0.0541, 0.0134, 0.0006, 0.0021, 0]
    # num200 = [0.2804, 0.1646, 0.0883, 0.0408, 0.0161, 0.0063, 0.0026, 0.0016]
    # num500 = [0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009]
    # num1000 = [0.7981, 0.6699, 0.3848, 0.164, 0.0619, 0.0232, 0.0078, 0.0028]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, delta1, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='$\eta = 1$')
    # plt.plot(ind, delta2, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='$\eta = 2$')
    # # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    # #          label='LF-GDPR')
    # plt.plot(ind, delta4, color='orange', alpha=1, linewidth=2, marker='o', markersize='7', label='$\eta = 4$')
    # plt.plot(ind, delta20, color='red', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    #          label='Naive RNL')
    # plt.plot(ind, rabv, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    # # plt.plot(ind, delta20, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    # #          label='SO-RNL')

    ground_true = 0.835
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, np.abs((ground_true - delta1) / ground_true), color='green', alpha=1, linestyle='--', linewidth=2,
             marker='x',
             markersize='7',
             label='update times=1 (SO-RNL)')
    plt.plot(ind, (ground_true - delta2) / ground_true, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s',
             markersize='7', label='update times=2 (SO-RNL)')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, (ground_true - delta4) / ground_true, color='orange', alpha=1, linestyle=':', linewidth=2, marker='s',
             markersize='7', label='update times=4 (SO-RNL)')
    plt.plot(ind, (ground_true - delta8) / ground_true, color='purple', alpha=1, linestyle=':', linewidth=2, marker='s',
             markersize='7', label='update times=8 (SO-RNL)')
    plt.plot(ind, np.abs((ground_true - ppdu4) / ground_true), color='red', alpha=1, linewidth=2, marker='o',
             markersize='7', label='update threshold=4 (PPDU)')

    plt.ylim(0, 1.0)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('Modularity Error', fontsize=17)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()
#
#
# Facebook_Mod_EdgeInser_times()

def Facebook_Gcc_EdgeInser_times():     # RNL的颜色统一为蓝色，但是
    delta1 = np.array([0.7993, 0.709, 0.6212, 0.3296, 0.1205, 0.0531, 0.034, 0.0281])
    delta2 = np.array([0.7976, 0.7707, 0.7584, 0.7091, 0.6634, 0.6308, 0.4923, 0.3149])
    # delta3 = [0.7338, 0.5282, 0.2919, 0.1282, 0.0502, 0.019, 0.0071, 0.0025]
    delta4 = np.array([0.7981, 0.7899, 0.7706, 0.7426, 0.7085, 0.671, 0.6299, 0.5886])
    delta8 = np.array([0.8308, 0.829, 0.8269, 0.8248, 0.8211, 0.8161, 0.8106, 0.8007])

    ppdu = np.array([0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009])
    ppdu4 = np.array([0.5842, 0.3364, 0.2223, 0.1882, 0.1586, 0.1322, 0.1069, 0.0845])

    # rabv =   [0.7443, 0.6761, 0.1665, 0.0541, 0.0134, 0.0006, 0.0021, 0]
    # num200 = [0.2804, 0.1646, 0.0883, 0.0408, 0.0161, 0.0063, 0.0026, 0.0016]
    # num500 = [0.5461, 0.291, 0.1171, 0.0443, 0.0168, 0.0055, 0.0025, 0.0009]
    # num1000 = [0.7981, 0.6699, 0.3848, 0.164, 0.0619, 0.0232, 0.0078, 0.0028]
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, delta1, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='$\eta = 1$')
    # plt.plot(ind, delta2, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='$\eta = 2$')
    # # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    # #          label='LF-GDPR')
    # plt.plot(ind, delta4, color='orange', alpha=1, linewidth=2, marker='o', markersize='7', label='$\eta = 4$')
    # plt.plot(ind, delta20, color='red', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    #          label='Naive RNL')
    # plt.plot(ind, rabv, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    # # plt.plot(ind, delta20, color='purple', alpha=1, linestyle='-.', linewidth=2, marker='x', markersize='7',
    # #          label='SO-RNL')

    # ground_true = 0.835
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, delta1, color='green', alpha=1, linestyle='--', linewidth=2,
             marker='x',
             markersize='7',
             label='update times=1 (SO-RNL)')
    plt.plot(ind, delta2, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s',
             markersize='7', label='update times=2 (SO-RNL)')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, delta4, color='orange', alpha=1, linestyle=':', linewidth=2, marker='s',
             markersize='7', label='update times=4 (SO-RNL)')
    plt.plot(ind, delta8, color='purple', alpha=1, linestyle=':', linewidth=2, marker='s',
             markersize='7', label='update times=8 (SO-RNL)')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o',
             markersize='7', label='update threshold=1 (PPDU)')
    # plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o',
    #          markersize='7', label='update threshold=4 (PPDU)')

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

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


Facebook_Gcc_EdgeInser_times()