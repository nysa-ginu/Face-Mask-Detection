import network_dev
import dataset_creation
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score


epochs = 50
train_loss_list = []
test_loss_list = []
counter = 0
total = 0
correct_predictions = 0
all_pred = []
all_tar = []


#training loop
for epoch in range(epochs):
    iteration = 0
    train_batch = iter(dataset_creation.train_data_loader)


    for data, target in train_batch:
        data = data.to(network_dev.device).float()
        target = target.to(network_dev.device).reshape(network_dev.batch_size,)


        network_dev.snn_net.train()
        spike_list, mem_list = network_dev.snn_net(data.view(network_dev.batch_size, -1))


        value_loss = torch.zeros((1), dtype=network_dev.dtype, device=network_dev.device)
        for step in range(network_dev.timesteps):
            value_loss += network_dev.loss(mem_list[step], target)


        network_dev.optimizer.zero_grad()
        value_loss.backward()
        network_dev.optimizer.step()


        train_loss_list.append(value_loss.item())

        #test set
        with torch.no_grad():
            network_dev.snn_net.eval()
            test_data, test_targets = next(iter(dataset_creation.test_data_loader))
            test_data = test_data.to(network_dev.device).float()
            test_targets = test_targets.to(network_dev.device).reshape(network_dev.batch_size,)
            test_spike, test_mem = network_dev.snn_net(test_data.view(network_dev.batch_size, -1))
            test_loss = torch.zeros((1), dtype=network_dev.dtype, device=network_dev.device)
            for step in range(network_dev.timesteps):
                test_loss += network_dev.loss(test_mem[step], test_targets)
            test_loss_list.append(test_loss.item())


            if iteration % 30 == 0:
                print(f"Epoch {epoch}, Iteration {iteration}")
                print(f"Train Set Loss: {train_loss_list[counter]:.2f}")
                print(f"Test Set Loss: {test_loss_list[counter]:.2f}")
                print("\n")
            counter += 1
            iteration +=1
            
#Visualizing Loss in train and test dataset
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

test_loader = dataset_creation.DataLoader(dataset_creation.test_dataset, batch_size=32, shuffle=True, drop_last=False)

#calculating the evaluation metric(accuracy, F1-score, Precision, Recall)
with torch.no_grad():
  network_dev.snn_net.eval()
  for data, target in test_loader:
    data = data.to(network_dev.device)
    target = target.to(network_dev.device)
    target = target.reshape(32,)
    test_spk, _ = network_dev.snn_net(data.float().view(data.size(0), -1))
    _, predicted = test_spk.sum(dim=0).max(1)
    total += target.size(0)
    correct_predictions += (predicted == target).sum().item()
    all_pred = all_pred + list(predicted)
    all_tar = all_tar + list(target)

print(f"Test Set Accuracy: {100 * correct_predictions / total:.2f}%")
print("f1 score: ",f1_score(all_tar,all_pred),"\nprecision: ",precision_score(all_tar,all_pred),"\nrecall: ",recall_score(all_tar,all_pred))