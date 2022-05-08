import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import torch


#creating custom dataset as the current dataset was not in the form used in Dataloader
class Dataset_custom(Dataset):
	def __init__(self,image_direc):
		self.image_path = image_direc
		files = glob.glob(self.image_path + "*")
		print(files)
		self.data = []
		for path_class in files:
			name_class = path_class.split("/")[-1]
			for image_path in glob.glob(path_class + "/*.png"):
				self.data.append([image_path, name_class])
		print(self.data)
		self.class_map = {"WithoutMask" : 0, "WithMask": 1}
		self.image_dimen = (32, 32)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, id):
		image_path, class_name = self.data[id]
		image = cv2.imread(image_path)
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image = cv2.resize(gray, self.image_dimen)
		id_class = self.class_map[class_name]
		image_tensor = torch.from_numpy(image)
		id_class = torch.tensor([id_class])
		return image_tensor, id_class
    
train_dataset = Dataset_custom("/Face Mask Dataset/Train/")
test_dataset = Dataset_custom("/Face Mask Dataset/Test/")


train_data_loader = DataLoader(train_dataset, batch_size=137, shuffle=True,drop_last=True)
test_data_loader = DataLoader(test_dataset, batch_size=137, shuffle=True,drop_last=True)