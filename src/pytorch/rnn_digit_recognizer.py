import torch
from torch import nn
from torch.autograd import Variable
from pytorch.digit_data import load_data_train


torch.manual_seed(1)


EPOCH = 10
BATCH_SIZE = 64
# rnn time step / image height
TIME_STEP = 28
# rnn input size / image width
INPUT_SIZE = 28
LR = 0.01

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

def train_model(data_trainX, data_trainY, data_testX, data_testY):
    """训练"""
    test_x = Variable(torch.Tensor(data_testX).float())
    test_y = Variable(torch.Tensor(data_testY).long())
    test_sum = test_y.size(0)
    for epoch in range(EPOCH):
        step = 0
        for i in range(0, len(data_trainX), BATCH_SIZE):
            step += 1
            b_x = Variable(torch.Tensor(data_trainX[i:i + BATCH_SIZE]).float().view(-1, 28, 28))
            b_y = Variable(torch.Tensor(data_trainY[i:i + BATCH_SIZE]).long())
            output = rnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                test_output = rnn(test_x.view(-1, 28, 28))
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy_sum = 0
                for i in range(test_sum):
                    if pred_y[i] == test_y[i]:
                        # print('预测:',pred_y[i], '正确:', y_test[i])
                        accuracy_sum += 1
                accuracy = accuracy_sum / test_sum
                print('Epoch: %s' % epoch)
                print('train loss: %s' % loss.data)
                print('test accuracy: %s' % accuracy)

if __name__ == '__main__':
    data_trainX, data_trainY, data_testX, data_testY = load_data_train()
    train_model(data_trainX, data_trainY, data_testX, data_testY)