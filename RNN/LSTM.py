import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size

        # 使用 LSTM 替换原先的 RNN
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # LSTM 输出包括隐藏状态和细胞状态
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self, batch_size):
        # 初始化隐藏状态和细胞状态
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


def char_tensor(string, char_to_idx):
    tensor = torch.zeros(1, len(string), len(char_to_idx))
    for c in range(len(string)):
        tensor[0, c, char_to_idx[string[c]]] = 1
    return tensor


def train(model, data, char_to_idx, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        hidden = model.init_hidden(1)
        for i in range(len(data) - 1):
            input_char = char_tensor(data[i], char_to_idx)
            target_char = torch.tensor([char_to_idx[data[i + 1]]], dtype=torch.long)

            optimizer.zero_grad()
            # 更新隐藏状态，保持细胞状态不变
            output, hidden = model(input_char, (hidden[0].detach(), hidden[1].detach()))
            loss = loss_fn(output, target_char)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f'Epoch {epoch} Loss: {total_loss / len(data)}')


def generate(model, char_to_idx, idx_to_char, start_str='tw', predict_len=100, temperature=0.85):
    model.eval()
    hidden = model.init_hidden(1)
    input_char = char_tensor(start_str, char_to_idx)
    predicted = start_str

    for p in range(predict_len):
        output, hidden = model(input_char[:, -1:, :], hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]

        predicted_char = idx_to_char[top_char.item()]
        predicted += predicted_char
        input_char = torch.cat((input_char, char_tensor(predicted_char, char_to_idx)[:, -1:, :]), 1)

    return predicted


# 数据和字符集
text = "twinkle, twinkle, little star, how I wonder what you are"
chars = sorted(set(text))

# 枚举字符，为每个字符分配一个唯一的索引
char_to_idx, idx_to_char = {}, {}
for idx, ch in enumerate(chars):
    char_to_idx[ch] = idx
    idx_to_char[idx] = ch

input_size = len(chars)
hidden_size = 128
output_size = len(chars)

model = CharLSTM(input_size, hidden_size, output_size)

train(model, text, char_to_idx)
print(generate(model, char_to_idx, idx_to_char, start_str='twinkle, twinkle, ', predict_len=200))
