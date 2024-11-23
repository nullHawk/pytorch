import torch
import model
from utils import line_to_tensor, load_data, N_LETTERS

def category_from_output(output):
    category_index = torch.argmax(output).item()
    return all_categories[category_index]

category_lines, all_categories = load_data()
rnn = model.RNN(N_LETTERS, 128, len(all_categories))
rnn.load_state_dict(torch.load('rnn4.pth'))
rnn.eval
while True:
    print('Enter a name:')
    line = input()
    if line == 'exit':
        break
    with torch.no_grad():
        input_tensor = line_to_tensor(line)
        hidden_tensor = rnn.init_hidden()
        for i in range(input_tensor.size()[0]):
            output, hidden_tensor = rnn(input_tensor[i], hidden_tensor)
        print(f"It is an {category_from_output(output)} name\n")
