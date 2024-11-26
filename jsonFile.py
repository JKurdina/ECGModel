import json
import random
import matplotlib.cm as cm
import numpy as np


def load(filename):
    with open(filename, 'r') as f:
        jsonFile = json.load(f)
    return jsonFile


def get_signals(file):
    signals = []
    for key in file:
        signals.append(file[key]['Leads'])
    return signals


def get_DelineationDoc(file):
    delDoc = []
    for key in file:
        delDoc.append(file[key]['Leads']['v1']['DelineationDoc'])
    return delDoc

class R_top:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def getSignalsAndDelDoc(file):
    signals = []
    delDoc = []
    for patient in file:
        for lead in file[patient]['Leads']:
            signals.append(file[patient]['Leads'][lead]['Signal'])
            points = []
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):

                point_R_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][1]
                point_R_y = file[patient]['Leads'][lead]['Signal'][point_R_x]
                point = R_top(point_R_x, point_R_y)
                points.append(point)
            delDoc.append(points)
    return signals, delDoc

def getSignalsAndDelDoc_Q(file):
    signals = []
    delDoc = []
    for patient in file:
        for lead in file[patient]['Leads']:
            signals.append(file[patient]['Leads'][lead]['Signal'])
            points = []
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                f = file[patient]['Leads'][lead]['DelineationDoc']['qrs']

                point_Q_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][0]
                point_Q_y = file[patient]['Leads'][lead]['Signal'][point_Q_x]
                point = R_top(point_Q_x, point_Q_y)
                points.append(point)
            delDoc.append(points)
    return signals, delDoc


def getSignalsAndDelDoc_S(file):
    signals = []
    delDoc = []
    for patient in file:
        for lead in file[patient]['Leads']:
            signals.append(file[patient]['Leads'][lead]['Signal'])
            points = []
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                f = file[patient]['Leads'][lead]['DelineationDoc']['qrs']

                point_S_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][2]
                point_S_y = file[patient]['Leads'][lead]['Signal'][point_S_x]
                point = R_top(point_S_x, point_S_y)
                points.append(point)
            delDoc.append(points)
    return signals, delDoc


def get_firstClass(file):
    signals = []
    points_R = []
    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                point_R_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][1]
                point_R_y = file[patient]['Leads'][lead]['Signal'][point_R_x]
                points_R.append(point_R_y)
                start = point_R_x - interval
                end = point_R_x + interval
                s = np.array(file[patient]['Leads'][lead]['Signal'][start:end])

                if len(s) != 500:
                    avg = np.mean(s)
                    while len(s) < 500:
                        s = np.append(s, avg)
                signals.append(s)
    return signals, points_R


def get_signals_small(file):

    signals = []
    points_R_small = []
    points_Q_small = []
    points_S_small = []
    points_P0_small = []
    points_P1_small = []
    points_P2_small = []
    for patient in file:
        for lead in file[patient]['Leads']:
            points_R_x = []
            points_R_y = []
            R_small = []

            points_Q_x = []
            points_Q_y = []
            Q_small = []

            points_S_x = []
            points_S_y = []
            S_small = []

            points_P0_x = []
            points_P0_y = []
            P0_small = []

            points_P1_x = []
            points_P1_y = []
            P1_small = []

            points_P2_x = []
            points_P2_y = []
            P2_small = []

            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                point_R_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][1]
                point_R_y = file[patient]['Leads'][lead]['Signal'][point_R_x]
                points_R_x.append(point_R_x)
                points_R_y.append(point_R_y)

                point_Q_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][0]
                point_Q_y = file[patient]['Leads'][lead]['Signal'][point_Q_x]
                points_Q_x.append(point_Q_x)
                points_Q_y.append(point_Q_y)

                point_S_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][2]
                point_S_y = file[patient]['Leads'][lead]['Signal'][point_S_x]
                points_S_x.append(point_S_x)
                points_S_y.append(point_S_y)

            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['p'])):

                point_P0_x = file[patient]['Leads'][lead]['DelineationDoc']['p'][i][0]
                point_P0_y = file[patient]['Leads'][lead]['Signal'][point_P0_x]
                points_P0_x.append(point_P0_x)
                points_P0_y.append(point_P0_y)

                point_P1_x = file[patient]['Leads'][lead]['DelineationDoc']['p'][i][1]
                point_P1_y = file[patient]['Leads'][lead]['Signal'][point_P1_x]
                points_P1_x.append(point_P1_x)
                points_P1_y.append(point_P1_y)

                point_P2_x = file[patient]['Leads'][lead]['DelineationDoc']['p'][i][2]
                point_P2_y = file[patient]['Leads'][lead]['Signal'][point_P2_x]
                points_P2_x.append(point_P2_x)
                points_P2_y.append(point_P2_y)

            s = file[patient]['Leads'][lead]['Signal']
            small_signal = [0] * (len(s)//3)

            k = 0
            for j in range(0, len(s)-3, 3):
                avg = (s[j] + s[j+1] + s[j+2])/3
                small_signal[k] = avg

                if j in points_R_x:
                    point = R_top(k, small_signal[k])
                    R_small.append(point)

                if j in points_Q_x:
                    point = R_top(k, small_signal[k])
                    Q_small.append(point)

                if j in points_S_x:
                    point = R_top(k, small_signal[k])
                    S_small.append(point)

                if j+1 in points_R_x:
                    point = R_top(k, small_signal[k])
                    R_small.append(point)

                if j+1 in points_Q_x:
                    point = R_top(k, small_signal[k])
                    Q_small.append(point)

                if j+1 in points_S_x:
                    point = R_top(k, small_signal[k])
                    S_small.append(point)

                if j+2 in points_R_x:
                    point = R_top(k, small_signal[k])
                    R_small.append(point)

                if j+2 in points_Q_x:
                    point = R_top(k, small_signal[k])
                    Q_small.append(point)

                if j+2 in points_S_x:
                    point = R_top(k, small_signal[k])
                    S_small.append(point)





                if j in points_P0_x:
                    point = R_top(k, small_signal[k])
                    P0_small.append(point)

                if j in points_P1_x:
                    point = R_top(k, small_signal[k])
                    P1_small.append(point)

                if j in points_P2_x:
                    point = R_top(k, small_signal[k])
                    P2_small.append(point)

                if j+1 in points_P0_x:
                    point = R_top(k, small_signal[k])
                    P0_small.append(point)

                if j+1 in points_P1_x:
                    point = R_top(k, small_signal[k])
                    P1_small.append(point)

                if j+1 in points_P2_x:
                    point = R_top(k, small_signal[k])
                    P2_small.append(point)

                if j+2 in points_P0_x:
                    point = R_top(k, small_signal[k])
                    P0_small.append(point)

                if j+2 in points_P1_x:
                    point = R_top(k, small_signal[k])
                    P1_small.append(point)

                if j+2 in points_P2_x:
                    point = R_top(k, small_signal[k])
                    P2_small.append(point)



                k+=1


            signals.append(small_signal)
            points_R_small.append(R_small)
            points_Q_small.append(Q_small)
            points_S_small.append(S_small)
            points_P0_small.append(P0_small)
            points_P1_small.append(P1_small)
            points_P2_small.append(P2_small)

    return signals, points_R_small, points_Q_small, points_S_small, points_P0_small, points_P1_small, points_P2_small

def get_firstClassSmall_R(signals, R_top):
    new_signals = []
    points_R = []
    interval = 83
    for i in range(len(R_top)):
        r = R_top[i]
        while(len(r) > 0):
            middle = r[0].x
            start = middle - interval
            end = middle + interval
            if (end > len(signals[i])):
                r = r[1:]
                break
            s = np.array(signals[i][start:end])
            new_signals.append(s)
            points_R.append(r[0].y)
            r = r[1:]

    return new_signals, points_R

def get_firstClassSmall_Q(signals, Q_top):
    new_signals = []
    points_Q = []
    interval = 83
    for i in range(len(Q_top)):
        r = Q_top[i]
        while(len(r) > 0):
            middle = r[0].x
            start = middle - interval
            end = middle + interval
            if (end > len(signals[i])):
                r = r[1:]
                break
            s = np.array(signals[i][start:end])
            new_signals.append(s)
            points_Q.append(r[0].y)
            r = r[1:]

    return new_signals, points_Q

def get_firstClassSmall_S(signals, S_top):
    new_signals = []
    points_S = []
    interval = 83
    for i in range(len(S_top)):
        r = S_top[i]
        while(len(r) > 0):
            middle = r[0].x
            start = middle - interval
            end = middle + interval
            if (end > len(signals[i])):
                r = r[1:]
                break
            s = np.array(signals[i][start:end])
            new_signals.append(s)
            points_S.append(r[0].y)
            r = r[1:]

    return new_signals, points_S

def get_firstClassSmall_S(signals, S_top):
    new_signals = []
    points_S = []
    interval = 83
    for i in range(len(S_top)):
        r = S_top[i]
        while(len(r) > 0):
            middle = r[0].x
            start = middle - interval
            end = middle + interval
            if (end > len(signals[i])):
                r = r[1:]
                break
            s = np.array(signals[i][start:end])
            new_signals.append(s)
            points_S.append(r[0].y)
            r = r[1:]

    return new_signals, points_S

def get_firstClass_Q(file):
    signals = []
    points_Q = []
    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                point_Q_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][0]
                point_Q_y = file[patient]['Leads'][lead]['Signal'][point_Q_x]
                points_Q.append(point_Q_y)
                start = point_Q_x - interval
                end = point_Q_x + interval
                s = np.array(file[patient]['Leads'][lead]['Signal'][start:end])

                if len(s) != 500:
                    avg = np.mean(s)
                    while len(s) < 500:
                        s = np.append(s, avg)
                signals.append(s)
    return signals, points_Q

def get_firstClass_S(file):
    signals = []
    points_S = []
    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                point_S_x = file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][2]
                point_S_y = file[patient]['Leads'][lead]['Signal'][point_S_x]
                points_S.append(point_S_y)
                start = point_S_x - interval
                end = point_S_x + interval
                s = np.array(file[patient]['Leads'][lead]['Signal'][start:end])

                if len(s) != 500:
                    avg = np.mean(s)
                    while len(s) < 500:
                        s = np.append(s, avg)
                signals.append(s)
    return signals, points_S

def get_firstClass_P1(file):
    signals = []
    points_P1 = []
    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['p'])):
                point_P1_x = file[patient]['Leads'][lead]['DelineationDoc']['p'][i][1]
                point_P1_y = file[patient]['Leads'][lead]['Signal'][point_P1_x]
                points_P1.append(point_P1_y)
                start = point_P1_x - interval
                end = point_P1_x + interval
                s = np.array(file[patient]['Leads'][lead]['Signal'][start:end])

                if len(s) != 500:
                    break
                signals.append(s)
    return signals, points_P1

def get_firstClass_T1(file):
    signals = []
    points_T1 = []
    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['t'])):
                point_T1_x = file[patient]['Leads'][lead]['DelineationDoc']['t'][i][1]
                point_T1_y = file[patient]['Leads'][lead]['Signal'][point_T1_x]
                points_T1.append(point_T1_y)
                start = point_T1_x - interval
                end = point_T1_x + interval
                s = np.array(file[patient]['Leads'][lead]['Signal'][start:end])

                if len(s) != 500:
                    break
                signals.append(s)
    return signals, points_T1

def getSignalsAndDelDoc_P1(file):
    signals = []
    delDoc = []
    for patient in file:
        for lead in file[patient]['Leads']:
            signals.append(file[patient]['Leads'][lead]['Signal'])
            points = []
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['p'])):
                f = file[patient]['Leads'][lead]['DelineationDoc']['p']

                point_P_x = file[patient]['Leads'][lead]['DelineationDoc']['p'][i][1]
                point_P_y = file[patient]['Leads'][lead]['Signal'][point_P_x]
                point = R_top(point_P_x, point_P_y)
                points.append(point)
            delDoc.append(points)
    return signals, delDoc

def getSignalsAndDelDoc_T1(file):
    signals = []
    delDoc = []
    for patient in file:
        for lead in file[patient]['Leads']:
            signals.append(file[patient]['Leads'][lead]['Signal'])
            points = []
            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['t'])):
                f = file[patient]['Leads'][lead]['DelineationDoc']['t']

                point_T_x = file[patient]['Leads'][lead]['DelineationDoc']['t'][i][1]
                point_T_y = file[patient]['Leads'][lead]['Signal'][point_T_x]
                point = R_top(point_T_x, point_T_y)
                points.append(point)
            delDoc.append(points)
    return signals, delDoc


def get_SecondClass(file):
    secondClass = []

    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            points_Q_x = []
            points_S_x = []

            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['qrs'])):
                points_Q_x.append(file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][0])
                points_S_x.append(file[patient]['Leads'][lead]['DelineationDoc']['qrs'][i][2])

            j = points_S_x[0] + 1
            while j < len(file[patient]['Leads'][lead]['Signal']):
                if len(points_Q_x) == 0 or len(points_S_x) == 0:
                    break
                if points_Q_x[0] <= j <= points_S_x[0]:
                    j = points_S_x[0] + 1
                    points_Q_x.pop(0)
                    points_S_x.pop(0)
                    continue
                else:
                    start = j - interval + 1
                    end = j + interval + 1
                    if end > 5000:
                        break
                    secondClass.append(np.array(file[patient]['Leads'][lead]['Signal'][start:end]))
                    j += 10



    return secondClass

def get_SecondClassSmall_R(signals, R_top, Q_top, S_top):
    new_signals = []
    interval = 83
    Q = Q_top
    S = S_top

    for i in range(len(signals)):
        if len(S[i]) == 0:
            break
        k = S[i][0].x + 1
        while k < len(signals[i]):
            if (len(new_signals) > 21954):
                break

            if len(Q[i]) == 0 or len(S[i]) == 0:
                break
            if Q[i][0].x <= k <= S[i][0].x:
                k = S[i][0].x + 1
                Q[i].pop(0)
                S[i].pop(0)
                continue
            else:
                start = k - interval + 1
                end = k + interval + 1
                if end > len(signals[i]):
                    break
                new_signals.append(np.array(signals[i][start:end]))
                k += 10


    return new_signals

def get_SecondClassSmall_P(signals, R_top, Q_top, S_top):
    new_signals = []
    interval = 83
    Q = Q_top
    S = S_top

    for i in range(len(signals)):
        k = S[i][0].x + 1
        while k < len(signals[i]):
            if (len(new_signals) > 21954):
                break

            if len(Q[i]) == 0 or len(S[i]) == 0:
                break
            if Q[i][0].x <= k <= S[i][0].x:
                k = S[i][0].x + 1
                Q[i].pop(0)
                S[i].pop(0)
                continue
            else:
                start = k - interval + 1
                end = k + interval + 1
                if end > len(signals[i]):
                    break
                new_signals.append(np.array(signals[i][start:end]))
                k += 10


    return new_signals


def get_SecondClass_P(file):
    secondClass = []

    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            points_P0_x = []
            points_P2_x = []

            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['p'])):
                points_P0_x.append(file[patient]['Leads'][lead]['DelineationDoc']['p'][i][0])
                points_P2_x.append(file[patient]['Leads'][lead]['DelineationDoc']['p'][i][2])

            if (len(points_P2_x)==0):
                break
            j = points_P2_x[0] + 1
            while j < len(file[patient]['Leads'][lead]['Signal']):
                if len(points_P0_x) == 0 or len(points_P2_x) == 0:
                    break
                if points_P0_x[0] <= j <= points_P2_x[0]:
                    j = points_P2_x[0] + 1
                    points_P0_x.pop(0)
                    points_P2_x.pop(0)
                    continue
                else:
                    start = j - interval + 1
                    end = j + interval + 1
                    if end > 5000:
                        break
                    secondClass.append(np.array(file[patient]['Leads'][lead]['Signal'][start:end]))
                    j += 10



    return secondClass

def get_SecondClass_T(file):
    secondClass = []

    interval = 250
    for patient in file:
        for lead in file[patient]['Leads']:
            points_T0_x = []
            points_T2_x = []

            for i in range(len(file[patient]['Leads'][lead]['DelineationDoc']['t'])):
                points_T0_x.append(file[patient]['Leads'][lead]['DelineationDoc']['t'][i][0])
                points_T2_x.append(file[patient]['Leads'][lead]['DelineationDoc']['t'][i][2])

            if (len(points_T2_x)==0):
                break
            j = points_T2_x[0] + 1
            while j < len(file[patient]['Leads'][lead]['Signal']):
                if len(points_T0_x) == 0 or len(points_T2_x) == 0:
                    break
                if points_T0_x[0] <= j <= points_T2_x[0]:
                    j = points_T2_x[0] + 1
                    points_T0_x.pop(0)
                    points_T2_x.pop(0)
                    continue
                else:
                    start = j - interval + 1
                    end = j + interval + 1
                    if end > 5000:
                        break
                    secondClass.append(np.array(file[patient]['Leads'][lead]['Signal'][start:end]))
                    j += 10



    return secondClass


def draw_signal(signal, ax):
    ax.plot(signal)
    ax.grid(True)
    ax.legend()


def draw_point(x, y, ax):
    cmap = cm.get_cmap('jet')
    color = cmap(random.random())
    ax.plot(x, y, marker='o', linestyle='', color=color)

def drawRtop(R_top, ax):
    cmap = cm.get_cmap('jet')
    color = cmap(random.random())
    for r in R_top:

        ax.plot(r.x, r.y, marker='o', linestyle='', color='red')

def drawXtop(R_top, ax):
    cmap = cm.get_cmap('jet')
    color = cmap(random.random())
    for r in R_top:
        ax.axvline(r.x, color='red', alpha=0.5)
        # ax.plot(r.x, 1, marker='o', linestyle='', color=color)


def draw_DelineationDoc(signal, delDoc, ax):
    cmap = cm.get_cmap('jet')
    for key in delDoc:
        draw_x = []
        draw_y = []
        for j in range(len(delDoc[key])):
            for x in range(len(delDoc[key][j]) - 1):
                draw_x.append(delDoc[key][j][x])
                draw_y.append(signal[delDoc[key][j][x]])
        color = cmap(random.random())
        ax.plot(draw_x, draw_y, marker='o', linestyle='', color=color, label=key)
