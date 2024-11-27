import matplotlib.pyplot as plt
import numpy as np
import torch
import jsonFile as jsonFile

if __name__ == '__main__':
    file = jsonFile.load("ecg_data_200.json")


    leads = ['i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'avf', 'avl', 'avr']
    tops = ['p', 'qrs', 't']

    # for lead in leads:
    #     for top in tops:
    #         save_model(lead, top)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 16))
    axes = [ax1, ax2, ax3]
    index = np.random.randint(0, 50, size=1)
    for i, top in enumerate(tops):
        avg = []
        for lead in leads:
            signals, points = jsonFile.getSignalsAndDelDoc(file, lead, top)
            signals = np.array(signals[150:])
            signals = torch.from_numpy(signals).float()
            original_signal = signals[index].numpy()

            model = torch.load(f"model_{top}_{lead}.pth")
            model.eval()


            votes = np.zeros(original_signal.shape[1])
            for p in range(0, original_signal.shape[1] - 500):
                binVote = model(signals[index][0][p:p + 500].unsqueeze(0))
                binVote = binVote.squeeze(0).detach().numpy()
                votes[p + 250 - 1] = binVote

            avg.append(votes)

        result = np.mean(avg, axis=0)
        jsonFile.draw_signal(result, axes[i])

    plt.savefig(f"Result1.png")
    plt.close()

