import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import os 
 
class LinearQNet(nn.Module) : 
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self,x) : 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x 

    def save(self, file_name='model.pth') : 
        model_folder_path = './model'
        # checking to see of the folder already exists 
        if not os.path.exists(model_folder_path) : 
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



# Class for actual training and optimization : 
class QTrainer : 
    def __init__(self, model, lr, gamma) :
        self.lr = lr 
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done)  :
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # If these already have multiple values (there are 2 kinds of calls to this function in the agent.py file)
        # Then these are already in the form of (n, x)
        # The (1, x) case is handled below as in there is only one value and we need to append one dimension 

        if len(state.shape) == 1 : 
            # (1,x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )         # just a tuple with one value

        # 1) Predicted 1 Values with the current state 
        prediction = self.model(state)


        # 2) Q_new =  reward + gamma  * max(next predicted Q value) -> only do this if not done
        #       > prediciton.clone()
        #       > preds[argmax(action)] = Q_new

        target = prediction.clone()
        for idx in range(len(done)) : 
            Q_new = reward[idx]
            if not done[idx] : 
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        
        # Creating the loss function  
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()

        

        
    