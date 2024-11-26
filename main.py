import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import jsonFile as jsonFile
from torch.utils.data import DataLoader
import Autocoder as ac
import ModelECG as model

if __name__ == '__main__':
    file = jsonFile.load("ecg_data_200.json")
    # signals = jsonFile.get_signals(file)  # 200 signals
    # delDoc = jsonFile.get_DelineationDoc(file)  # 200 doc

    # R_top
    # small_signals, R_top_small, Q_top_small, S_top_small, a, b, c = jsonFile.get_signals_small(file)
    # data_small_signals, data_points_R_small = jsonFile.get_firstClassSmall_R(small_signals, R_top_small)
    # secondClass_small_R = jsonFile.get_SecondClassSmall_R(small_signals, R_top_small, Q_top_small, S_top_small)
    # signals = np.array(small_signals)


    # small_signals, R_top_small, Q_top_small, S_top_small, P0_top_small, P1_top_small, P2_top_small = jsonFile.get_signals_small(file)
    # data_small_signals, data_points_R_small = jsonFile.get_firstClassSmall_S(small_signals, P1_top_small)
    # secondClass_small_R = jsonFile.get_SecondClassSmall_R(small_signals, P1_top_small, P0_top_small, P2_top_small)
    # signals = np.array(small_signals)

    element = 'QRS'
    QRS1_class1 = torch.load('train_class1_' + element + '_i_.pt', weights_only=True).float()
    QRS1_class2 = torch.load('train_class2_' + element + '_i_.pt', weights_only=True)[:QRS1_class1.size()[0]].float()


    # signals, R_top = jsonFile.getSignalsAndDelDoc(file)


    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))
    # jsonFile.draw_signal(secondClass_small_R[5], ax1)
    # # jsonFile.drawRtop(Q_top_small[5], ax1)
    # # ax1.plot(R_top_small[5].x, R_top_small[5].y, marker='o', linestyle='', color='red')
    #
    # jsonFile.draw_signal(secondClass_small_R[10], ax2)
    # # jsonFile.drawRtop(S_top_small[10], ax2)
    # # ax1.plot(R_top_small[10].x, R_top_small[10].y, marker='o', linestyle='', color='red')
    # plt.savefig(f"R_top_{5}.png")
    # plt.close()

    # Обучение модели на R-пике
    # data_signals, data_points_R = jsonFile.get_firstClass(file)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))
    # jsonFile.draw_signal(data_signals[5], ax1)
    # ax1.plot(data_points_R[5], marker='o', linestyle='', color='red')
    # jsonFile.draw_signal(data_signals[10], ax2)
    # ax2.plot(data_points_R[10], marker='o', linestyle='', color='red')
    # plt.savefig(f"R_top_{5}.png")
    # plt.close()


    # secondClass = jsonFile.get_SecondClass(file)

    # data_signals_small, data_points_R_small = jsonFile.get_firstClassSmall_R(file)
    #
    # signals, R_top = jsonFile.getSignalsAndDelDoc(file)



    # # Обучение модели на Q-пике
    # data_signals, data_points_Q = jsonFile.get_firstClass_Q(file)
    # secondClass = jsonFile.get_SecondClass(file)
    #
    # signals, Q_top = jsonFile.getSignalsAndDelDoc_Q(file)

    # # Обучение модели на S-пике
    # data_signals, data_points_S = jsonFile.get_firstClass_S(file)
    # secondClass = jsonFile.get_SecondClass(file)

    # signals, S_top = jsonFile.getSignalsAndDelDoc_S(file)

    # # Обучение модели на P1-пике
    # data_signals, data_points_R = jsonFile.get_firstClass_P1(file)
    # secondClass = jsonFile.get_SecondClass_P(file)
    #
    # signals, P1_top = jsonFile.getSignalsAndDelDoc_P1(file)

    # # Обучение модели на T1-пике
    # data_signals, data_points_R = jsonFile.get_firstClass_T1(file)
    # secondClass = jsonFile.get_SecondClass_T(file)
    #
    # signals, T1_top = jsonFile.getSignalsAndDelDoc_T1(file)
    # signals = np.array(signals)



    # # Отрисовка ЭКГ
    #
    # for i in range(0, 500, 100):
    #     fig, ax = plt.subplots(figsize=(12, 4))
    #     jsonFile.draw_signal(SecondClass[i], ax)
    #     ax.legend()
    #     plt.plot(250, SecondClass[i][250], linestyle='', marker='*')
    #     plt.show()

    # # # Создание наборов данных
    # train_data = np.array(data_signals[:17500])
    # test_data = np.array(data_signals[17500:])
    #
    # train_data_2 = np.array(secondClass[:17500])
    # test_data_2 = np.array(secondClass[17500:])

    size = 15000


    # # # Создание наборов данных
    # train_data = np.array(data_small_signals[:size])
    # test_data = np.array(data_small_signals[size:])
    #
    # train_data_2 = np.array(secondClass_small_R[:size])
    # test_data_2 = np.array(secondClass_small_R[size:])
    #
    # # signals = np.array(signals)
    #
    # train_data = torch.from_numpy(train_data).float()
    # test_data = torch.from_numpy(test_data).float()
    #
    # train_data_2 = torch.from_numpy(train_data_2).float()
    # test_data_2 = torch.from_numpy(test_data_2).float()
    #
    # signals = torch.from_numpy(signals).float()

    # bin_vote = torch.zeros(17500, 500)
    # for i in range(17500):
    #     bin_vote[i][249] = 1
    # bin_vote = DataLoader(bin_vote, batch_size=50, shuffle=True)
    #
    # bin_vote_test = torch.zeros(4466, 500)
    # for i in range(4466):
    #     bin_vote_test[i][249] = 1

    # bin_vote = torch.zeros(size, 500)
    # for i in range(size):
    #     bin_vote[i][249] = 1
    # bin_vote = DataLoader(bin_vote, batch_size=50, shuffle=True)
    #
    # train_loader = DataLoader(train_data, batch_size=50, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=50, shuffle=False)
    #
    # train_loader_2 = DataLoader(train_data_2, batch_size=50, shuffle=True)

    # bin_vote = torch.zeros(size, 166)
    # for i in range(size):
    #     bin_vote[i][83-1] = 1
    # bin_vote = DataLoader(bin_vote, batch_size=50, shuffle=True)

    train_loader = DataLoader(QRS1_class1, batch_size=50, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=50, shuffle=False)

    train_loader_2 = DataLoader(QRS1_class2, batch_size=50, shuffle=False)

    # model = ac.Autoencoder()
    model = model.ModelECG()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.MSELoss()
    losses = []


    coeff = torch.full((50, 166), 0.1)

    for m in range(coeff.shape[0]):
        for n in range(coeff.shape[1]):
            if 125 <= n <= 248 or 253 <= n <= 375:
                coeff[m][n] = 0.5
            if 248 <= n <= 253:
                coeff[m][n] = 1

    # for i in range(248, 253):
    #     coeff[i] = 1
    #
    # for i in range(125, 248):
    #     coeff[i] = 0.3
    #
    # for i in range(253, 375):
    #     coeff[i] = 0.3

    coeff2 = torch.zeros((50, 166))

    # target1 = torch.zeros(64, 1)
    # for i in range(len(target1)):
    #     target1[i] = 1
    # target2 = torch.zeros(64, 1)

    losses1 = []
    losses2 = []
    # Обучение
    for epoch in range(1):
        for i, (data_batch_1, data_batch_2) in enumerate(zip(train_loader, train_loader_2)):
            optimizer.zero_grad()
            # Обучаем модель на батче из train_loader
            outputs_1, sigm_1 = model(data_batch_1)




            if(data_batch_1.shape[0] == 50):
                outputs_1 = outputs_1 * coeff
                data_batch_1 = data_batch_1 * coeff
            else:
                c = torch.full((len(data_batch_1), 166), 0.1)
                for m in range(c.shape[0]):
                    for n in range(c.shape[1]):
                        if 125 <= n <= 248 or 253 <= n <= 375:
                            c[m][n] = 0.5
                        if 248 <= n <= 253:
                            c[m][n] = 1
                outputs_1 = outputs_1 * c
                data_batch_1 = data_batch_1 * c

            loss_data_1 = criterion(outputs_1, data_batch_1)
            target1 = torch.ones(data_batch_1.shape[0], 1)
            loss_sigm_1 = criterion(sigm_1, target1)
            loss_1 = loss_data_1 + loss_sigm_1
            loss_1.backward()

            # Обучаем модель на батче из train_loader_2
            outputs_2, sigm_2 = model(data_batch_2)
            loss_data_2 = criterion(outputs_2, data_batch_2) * 0
            target2 = torch.zeros(data_batch_2.shape[0], 1)
            loss_sigm_2 = criterion(sigm_2, target2)
            loss_2 = loss_data_2 + loss_sigm_2
            loss_2.backward()

            optimizer.step()

        print('Epoch:', epoch, 'Loss:', loss_1.item(), loss_2.item())
        print('Loss binary vote:', loss_sigm_1.item(), loss_sigm_2.item())
        losses1.append(loss_1.item())
        losses2.append(loss_2.item())
    # losses.append(loss.item())
    plt.figure(figsize=(20, 10))
    plt.plot(losses1)
    plt.title("Loss1")
    plt.savefig("Loss1.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(losses2)
    plt.title("Loss2")
    plt.savefig("Loss2.png")
    plt.close()

    # # Тестирование
    # model.eval()  # Переводим модель в режим оценки
    # with torch.no_grad():
    #     test_loss = 0
    #     test_loss_bin_vote = 0
    #     for data_batch in test_loader:
    #         outputs, sigm = model(data_batch)
    #         print(sigm)
    #         test_loss += criterion(outputs, data_batch).item()
    #         test_loss_bin_vote += criterion(sigm, bin_vote_test).item()
    #
    #     test_loss /= len(test_loader)
    #     test_loss_bin_vote /= len(test_loss_bin_vote)
    #     print('Test Loss:', test_loss)
    #     print('Test Loss bin:', test_loss_bin_vote)

    model.eval()

    indexes = np.random.randint(0, len(signals), size=10)

    for i, index in enumerate(indexes):
        original_signal = signals[index].numpy()
        votes = np.zeros(1666)
        for p in range(0, len(original_signal)):

            decoded_signal, binVote = model(signals[index][p:p + 166-1].unsqueeze(0))

            decoded_signal = decoded_signal.squeeze(0).detach().numpy()
            binVote = binVote.squeeze(0).detach().numpy()
            votes[p + 83 - 1] = binVote

            if p + 166 == len(original_signal):
                break

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))
        jsonFile.draw_signal(original_signal, ax1)
        # jsonFile.drawRtop(P1_top_small[index], ax1)
        ax2.plot(votes)
        ax2.grid(True)
        # jsonFile.drawXtop(P1_top_small[index], ax2)
        plt.savefig(f"P_top_{i}.png")
        plt.close()

    # # Отрисовка ЭКГ
    # indexes = np.random.randint(0, len(test_data), size=10)
    #
    # for i, index in enumerate(indexes):
    #     original_signal = test_data[index].numpy()
    #
    #     decoded_signal, binVote = model(test_data[index].unsqueeze(0))
    #
    #     decoded_signal = decoded_signal.squeeze(0).detach().numpy()
    #     binVote = binVote.squeeze(0).detach().numpy()
    #
    #
    #     # R_top_x = []
    #     # R_top_y = []
    #     # for r in range(len(s)):
    #     #     if s[r] == 1:
    #     #         R_top_y.append(original_signal[r])
    #     #         R_top_x.append(r)
    #
    #
    #     plt.figure(figsize=(20, 10))
    #
    #     plt.plot(original_signal, color='grey', alpha=0.5, label='Исходный сигнал')
    #     plt.title(f"Cигнал {i + 1}, binVote: {binVote}")
    #
    #     plt.plot(decoded_signal, color='blue', label='Декодированный сигнал')
    #     # plt.plot(R_top_x, R_top_y, color='red', label='R-top', linestyle='', marker='*')
    #
    #     plt.tight_layout()
    #     plt.savefig(f"ecg_signal_{i + 1}.png")
    #     plt.close()
    #
    #     # Отрисовка ЭКГ
    # indexes = np.random.randint(0, len(test_data_2), size=10)
    #
    # for i, index in enumerate(indexes):
    #     original_signal_2 = test_data_2[index].numpy()
    #
    #     decoded_signal_2, binVote2 = model(test_data_2[index].unsqueeze(0))
    #
    #
    #     decoded_signal_2 = decoded_signal_2.squeeze(0).detach().numpy()
    #     binVote2 = binVote2.squeeze(0).detach().numpy()
    #
    #
    #     plt.figure(figsize=(20, 10))
    #
    #     plt.plot(original_signal_2, color='grey', alpha=0.5, label='Исходный сигнал')
    #     plt.title(f"Cигнал {i + 1}, binVote: {binVote2}")
    #
    #     plt.plot(decoded_signal_2, color='green', label='Декодированный сигнал')
    #         # plt.plot(R_top_x, R_top_y, color='red', label='R-top', linestyle='', marker='*')
    #
    #     plt.tight_layout()
    #     plt.savefig(f"2_ecg_signal_{i + 1}.png")
    #     plt.close()
    #
    # # # Отрисовка ЭКГ
    # # fig, ax = plt.subplots(figsize=(12, 4))
    # # jsonFile.draw_signal(data_signals[10], ax)
    # # jsonFile.draw_point(249, data_points_R[10], ax)
    # # ax.legend()
    # # plt.show()
    #
    # for i in range(10):
    #     # Отрисовка ЭКГ
    #     fig, ax = plt.subplots(figsize=(12, 4))
    #     jsonFile.draw_signal(signals[i], ax)
    #     jsonFile.drawRtop(R_top[i], ax)
    #     ax.legend()
    #     plt.show()
