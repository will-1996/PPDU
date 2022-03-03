import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FuncFormatter
# import pandas as pd
import copy
from matplotlib.pyplot import MultipleLocator



sizeFace = 4039
sizeEnron = 36692
sizeAstro = 18772

def Facebook_GccError():
    rabv = [0.7443, 0.6761, 0.1665, 0.0541, 0.0134, 0.0006, 0.0021, 0]
    dgg = [0.8904, 0.8905, 0.892, 0.8918, 0.8894, 0.8901, 0.8904, 0.891]
    ppdu = [0.5032, 0.2682, 0.1176, 0.0492, 0.0181, 0.0065, 0.0028, 0.001]
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

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('GCC Error', fontsize=17)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Facebook_GccError()

def Facebook_ccError():
    rabv = [0.836, 0.7826, 0.5094, 0.1488, 0.0544, 0.0014, 0.0008, 0.0002]
    dgg = [0.8997, 0.8955, 0.9001, 0.8964, 0.8986, 00.8988, 0.8978, 0.8967]
    ppdu = [0.6845, 0.4767, 0.2658, 0.1188, 0.053, 0.0196, 0.0085, 0.0028]
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
    rabv = np.array([0.97, 0.95, 0.92, 0.88, 0.80, 0.63, 0.32, 0.14])
    dgg = np.array([-0.0014, 0.001, 0.0019, -0.0011, -0.0005, 0.0003, -0.0009, -0.0014])
    ppdu = np.array([0.9125, 0.8286, 0.8341, 0.8347, 0.8348, 0.8357, 0.8349, 0.8349])
    ind = np.arange(8) + 1

    ground_true = 0.835
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x',
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



def Facebook_ARI():
    rabv = np.array([0.2669, 0.4438, 0.8047, 0.8937, 0.9632, 0.987, 0.991, 0.9858])
    dgg = np.array([0.0001, 0.0004, 0, 0.0001, 0.0003, 0.0013, 0.0004, -0.0002])
    ppdu = np.array([0.8286, 0.9125, 0.9397, 0.9493, 0.9783, 0.9928, 0.9954, 0.9912])
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=17)
    plt.ylabel('ARI', fontsize=17)

    plt.legend(loc='upper right', fontsize=17)
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Facebook_ARI()

def Facebook_AMI():
    rabv = np.array([0.3649, 0.6191, 0.8356, 0.9347, 0.9572, 0.9837, 0.988, 0.99])
    dgg = np.array([-0.0004, -0.0005, 0.0011, -0.001, 0.0005, 0.0014, 0.0014, -0.0006])
    ppdu = np.array([0.8374, 0.9049, 0.9459, 0.9551, 0.976, 0.992,  0.9953, 0.9911])
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('AMI', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Facebook_AMI()


def faceBook_ARMI():
    ari_lf = np.array([0.2669, 0.4438, 0.8047, 0.8937, 0.9632, 0.987, 0.991, 0.9858])+0.2
    ami_lf = np.array([0.3649, 0.6191, 0.8356, 0.9347, 0.9572, 0.9837, 0.988, 0.99])+0.2
    ari_ldpg = np.array([0.0001, 0.0004, 0, 0.0001, 0.0003, 0.0013, 0.0004, -0.0002])+0.2
    ami_ldpg = np.array([-0.0004, -0.0005, 0.0011, -0.001, 0.0005, 0.0014, 0.0014, -0.0006])+0.2
    ari_wdt = np.array([0.8286, 0.9125, 0.9397, 0.9493, 0.9783, 0.9928, 0.9954, 0.9912])+0.2
    ami_wdt = np.array([0.8374, 0.9049, 0.9459, 0.9551, 0.976, 0.992,  0.9953, 0.9911])+0.2

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

    # lns4 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_ldpg, color='#4DBBD5CC', label='AMI(LDPGen)', linewidth=2,
    #                 marker='s', markersize='7')
    #
    # lns5 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_wdt, color='#F39B7FCC', label='AMI(Wdt-SCAN)', linewidth=2,
    #                 marker='o', markersize='7')
    #
    # lns6 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_lf, color='#00A087CC', label='AMI(LF-GDPR)', linewidth=2,
    #                 marker='^', markersize='7')




    fig.legend(loc=2, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
    # fig.legend(loc=(0.65, 0.11))
    # ax.grid()
    ax.set_xlabel('Privacy Budget')
    ax.set_ylabel('ARI')
    ax2.set_ylabel('AMI')

    my_y_ticks = np.arange(-0.2, 1.2, 0.2)
    y_major_locator = MultipleLocator(0.2)
    # ax.yticks(my_y_ticks)
    # ax.set_ylim(-0.2, 1.2, ('100', '200', '300', '400', '500'))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])

    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax2.set_yticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    # ax2.set_ylim(-0.2, 1.2)
    plt.show()

# faceBook_ARMI()



# ------------------------------Enron -------------------------------------

def Enron_GccError():
    rabv = [0.9846, 0.9654, 0.8388, 0.424, 0.0494, 0.0061, 0.001, 0]
    dgg = [0.6703, 0.6666, 0.6592, 0.6558, 0.6599, 0.6579, 0.6618, 0.6634]
    ppdu = [0.4363, 0.2087, 0.0933, 0.037, 0.0133, 0.0047, 0.0022, 0.0007]
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

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('GCC Error', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Enron_GccError()

def Enron_ccError():
    rabv = [0.995, 0.99, 0.9861, 0.9523, 0.6381, 0.157, 0.0233, 0.003]
    dgg = [0.9329, 0.9298, 0.9323, 0.9324, 0.9326, 0.931, 0.9328, 0.933]
    ppdu = [0.707, 0.4367, 0.2102, 0.0893, 0.0326, 0.0124, 0.0044, 0.0016]
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

# Enron_ccError()


def Enron_Mod():
    # rabv = np.array([0.6749, 0.7961, 0.7936, 0.8344, 0.835, 0.8338, 0.8349, 0.8349])
    # rabv = np.array([0.3653, 0.6724, 0.7626, 0.7553, 0.8251, 0.8309, 0.8333, 0.8339])
    rabv = np.array([0.003, 0.0326, 0.0739, 0.2359, 0.4702, 0.5943, 0.6023, 0.618])
    dgg = np.array([0.2432, 0.2432, 0.2442, 0.2439, 0.2443, 0.2442, 0.2445, 0.2437])
    ppdu = np.array([0.4022, 0.4857, 0.5634, 0.583, 0.6097, 0.6174, 0.6054, 0.6179])
    ind = np.arange(8) + 1

    ground_true = 0.618
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x',
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

# Enron_Mod()

# to be continued-------------------------------------------------

def Enron_ARI():
    rabv = np.array([0.005, 0.0112, 0.0133, 0.0179, 0.0198, 0.0567, 0.4299, 0.578])
    dgg = np.array([0.0014, 0.0021, 0.0015, 0.0002, 0.002, 0.0017, 0.0017, 0.0007])
    ppdu = np.array([0.2074, 0.2604, 0.4227, 0.4935, 0.6266, 0.6305, 0.6512, 0.6784])
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('ARI', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Enron_ARI()


def Enron_AMI():
    rabv = np.array([0.005, 0.011, 0.027, 0.0901, 0.289, 0.617, 0.683, 0.776])
    dgg = np.array([0.0019, 0.0026, 0.0023, 0.0014, 0.0031, 0.0015, 0.0014, 0.0014])
    ppdu = np.array([0.313, 0.4076, 0.5842, 0.6256, 0.7609, 0.7856, 0.7965, 0.8129])
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('AMI', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Enron_AMI()
def Enron_ARMI():
    ari_lf = np.array([0.005, 0.0112, 0.0133, 0.0179, 0.0198, 0.0567, 0.4299, 0.578])+0.2
    ami_lf = np.array([0.005, 0.011, 0.027, 0.0901, 0.289, 0.617, 0.683, 0.776])+0.2
    ari_ldpg = np.array([0.0014, 0.0021, 0.0015, 0.0002, 0.002, 0.0017, 0.0017, 0.0007])+0.2
    ami_ldpg = np.array([0.0019, 0.0026, 0.0023, 0.0014, 0.0031, 0.0015, 0.0014, 0.0014])+0.2
    ari_wdt = np.array([0.2074, 0.2604, 0.4227, 0.4935, 0.6266, 0.6305, 0.6512, 0.6784])+0.2
    ami_wdt = np.array([0.313, 0.4076, 0.5842, 0.6256, 0.7609, 0.7856, 0.7965, 0.8129])+0.2

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

    # lns4 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_ldpg, color='#4DBBD5CC', label='AMI(LDPGen)', linewidth=2,
    #                 marker='s', markersize='7')
    #
    # lns5 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_wdt, color='#F39B7FCC', label='AMI(Wdt-SCAN)', linewidth=2,
    #                 marker='o', markersize='7')
    #
    # lns6 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_lf, color='#00A087CC', label='AMI(LF-GDPR)', linewidth=2,
    #                 marker='^', markersize='7')




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
    ax.set_yticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax2.set_yticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    # ax2.set_ylim(-0.2, 1.2)
    plt.show()

# Enron_ARMI()

# ----------------------------Astro------------------------------

def Astro_GccError():
    rabv = [0.9762, 0.9423, 0.8827, 0.3325, 0.0519, 0.0066, 0.0009, 0.0001]
    dgg = [0.9672, 0.9679, 0.9675, 0.9676, 0.9678, 0.9672, 0.9675, 0.9679]
    ppdu = [0.5128, 0.267, 0.1167, 0.0458, 0.0174, 0.0068, 0.0026, 0.0009]
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

# Astro_GccError()

def Astro_ccError():
    rabv = [0.992, 0.9817, 0.9613, 0.772, 0.3323, 0.062, 0.0072, 0.001]
    dgg = [0.985, 0.9839, 0.9843, 0.9848, 0.984, 0.984, 0.9838, 0.9842]
    ppdu = [0.7596, 0.5286, 0.2828, 0.1241, 0.05, 0.02, 0.0073, 0.0028]
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

# Astro_ccError()


def Astro_Mod():
    # rabv = np.array([0.6749, 0.7961, 0.7936, 0.8344, 0.835, 0.8338, 0.8349, 0.8349])
    # rabv = np.array([0.3653, 0.6724, 0.7626, 0.7553, 0.8251, 0.8309, 0.8333, 0.8339])
    rabv = np.array([0.031, 0.084, 0.1407, 0.3992, 0.5804, 0.6197, 0.6249, 0.6258])
    dgg = np.array([0.1676, 0.1688, 0.17, 0.1688, 0.1681, 0.1699, 0.1712, 0.1703])
    ppdu = np.array([0.397, 0.5067, 0.5754, 0.6038, 0.6211, 0.6244, 0.629, 0.6219])
    ind = np.arange(8) + 1

    ground_true = 0.623
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    # plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
    #          label='SO-RNL')
    plt.plot(ind, (ground_true - rabv) / ground_true, color='green', alpha=1, linestyle='--', linewidth=2, marker='x',
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

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('Modularity Error', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Astro_Mod()



def Astro_ARI():
    rabv = np.array([0.0122, 0.0211, 0.0375, 0.1773, 0.311, 0.339, 0.343, 0.385])
    dgg = np.array([0.0002, 0.0006, 0.0009, 0.0005, 0.0005, 0.0007, 0.0002, 0.0007])
    ppdu = np.array([0.1872, 0.27,  0.365, 0.391, 0.415, 0.4214, 0.428, 0.467])
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('ARI', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()


# Astro_ARI()

def Astro_AMI():
    rabv = np.array([0.0533, 0.0724, 0.1609, 0.3222, 0.47, 0.53, 0.577, 0.602])
    dgg = np.array([0.0041, 0.0029, 0.0031, 0.0022, 0.0019, 0.0026, 0.0012, 0.0025])
    ppdu = np.array([0.322, 0.4348, 0.5582, 0.5677, 0.585, 0.601, 0.617, 0.634])
    ind = np.arange(8) + 1
    # Draw Plot
    # plt.figure(figsize=(16,10), dpi= 80)

    plt.plot(ind, rabv, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='SO-RNL')
    plt.plot(ind, dgg, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='DDGU')
    # plt.plot(ind, ppdu, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    plt.plot(ind, ppdu, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='PPDU')
    plt.ylim(-0.2, 1.2)
    plt.xlim(1, 8)

    ax = plt.gca()
    # plt.gca().set_xticks(np.arange(1, 8.5, 0.5))

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlabel('Privacy Budget', fontsize=15)
    plt.ylabel('AMI', fontsize=15)

    plt.legend(loc='upper right')
    plt.grid()
    plt.grid(axis='both', alpha=.4, which='both')
    plt.show()

# Astro_AMI()


def faceBook_ARMI():
    ari_lf = np.array([0.13, 0.23, 0.37, 0.5, 0.6, 0.7, 0.9, 0.9])
    ami_lf = np.array([0.08, 0.25, 0.41, 0.53, 0.62, 0.73, 0.89, 0.9])
    ari_ldpg = np.array([0.2, 0.21, 0.23, 0.25, 0.28, 0.31, 0.33, 0.35])
    ami_ldpg = np.array([0.14, 0.2, 0.23, 0.25, 0.32, 0.37, 0.39, 0.4])
    ari_wdt = np.array([0.43, 0.45, 0.81, 0.90, 0.91, 0.93, 0.98, 0.99])
    ami_wdt = np.array([0.69, 0.72, 0.87, 0.92, 0.91, 0.93, 0.96, 0.98])

    width = 0.8 / 3

    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.bar(np.arange(len(ari_lf)) + 1 - width, ari_wdt, width=0.8 / 3, label='ARI(Wdt-SCAN)')
    lns2 = ax.bar(np.arange(len(ari_lf)) + 1 + width - width, ari_lf, width=0.8 / 3, label='ARI(LF-GDPR)')
    lns3 = ax.bar(np.arange(len(ari_lf)) + 1 + 2 * width - width, ari_ldpg, width=0.8 / 3, label='ARI(LDPGen)')
    ax2 = ax.twinx()
    lns4 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_wdt, label='AMI(Wdt-SCAN)', linewidth=2, marker='o', markersize='7')
    lns5 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_lf, label='AMI(LF-GDPR)', linewidth=2, marker='^', markersize='7')
    lns6 = ax2.plot(np.arange(len(ari_lf)) + 1, ami_ldpg, label='AMI(LDPGen)', linewidth=2, marker='s', markersize='7')

    fig.legend(loc=2, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
    # ax.grid()
    ax.set_xlabel('Privacy Budget')
    ax.set_ylabel('ARI')
    ax2.set_ylabel('AMI')
    plt.show()


# faceBook_ARMI()