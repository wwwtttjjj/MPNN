import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.backends import cudnn
from dataloaders.dataset import TwoStreamBatchSampler
from utils.dataset_loader_cvpr import MyData
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from networks.DD import UnFNet_singal
from utils import dice_score
import torch.nn.functional as F
import funcy
import torch.backends.cudnn as cudnn
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
import os
from utils import ramps
import random
from PIL import Image
import bin_map
from torch.utils.data.sampler import SubsetRandomSampler
from test import test_model
BM = [bin_map.bin3,bin_map.bin4,bin_map.bin5]
os.environ["WANDB_MODE"] = "dryrun"
import pymic

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
fs_observer = os.path.join(BASE_PATH, "MICAD_results")
if not os.path.exists(fs_observer):
	os.makedirs(fs_observer)
np.set_printoptions(threshold=np.inf)

parameters = dict(
		max_iteration=80000,
		spshot=30,
		nclass=2,
		batch_size=8,
		sshow=655,
		phase="train",  # train or test
		param=False,  # Loading checkpoint
		dataset="Magrabia",  # test or val (dataset)
		snap_num=20,  # Snapshot Number
		gpu_ids='0',  # CUDA_VISIBLE_DEVICES
)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=parameters["gpu_ids"], type=str, help='gpu device ids')
parser.add_argument('--seed', default=1337, type=int, help='manual seed')
parser.add_argument('--arch', default='resnet34', type=str, help='backbone model name')
parser.add_argument('--batch_size', default=parameters["batch_size"], type=int, help='batch size for train')
parser.add_argument('--phase', default=parameters["phase"], type=str, help='train or test')
parser.add_argument('--param', default=parameters["param"], type=str, help='path to pre-trained parameters')

parser.add_argument('--train_dataroot', default='./DiscRegion', type=str, help='path to train data')
parser.add_argument('--test_dataroot', default='./DiscRegion', type=str, help='path to test or val data')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--val_root', default='./Out/val', type=str, help='directory to save run log')
parser.add_argument('--log_root', default='./Out/log', type=str, help='directory to save run log')
parser.add_argument('--snapshot_root', default='./Out/snapshot', type=str, help='path to checkpoint or snapshot')
parser.add_argument('--output_root', default='./Out/results', type=str, help='path to saliency map')
parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
parser.add_argument("--patience", type=int, default=100, help="最大容忍不变epoch")
parser.add_argument('--label_unlabel', type=str, default='MCPLD-70-585', help='GPU to use')
parser.add_argument("--max_iterations", type=int, default=parameters["max_iteration"], help="maxiumn epoch to train")
#############
parser.add_argument('--consistency_type', type=str,
					default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
					default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
					default=200.0, help='consistency_rampup')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--base_lr', type=float, default=0.00005,
					help='segmentation network learning rate')
parser.add_argument('--Un', type=int, default=1,
					help='Un or not Un')
parser.add_argument('--val_num', type=int, default=0,
					help='the subset as val_set')

parser.add_argument('--number', type=int, default=5,
					help='3-5')

def ave(label_batch):
	most_frequent_values, _ = torch.mode(torch.stack(label_batch), dim=0)
	return most_frequent_values


def split_test(val_num):
    all_test = [i for i in range(0, 95)]
    # val_set = all_test[val_num * 19:(val_num + 1)*19]
    # test_set = all_test[0:val_num*19] + all_test[(val_num+1)*19:]
    val_set = all_test
    test_set = all_test
    
    return val_set,test_set
def ValTest(parameters):
    val_list, test_list = split_test(args.val_num)

    val_sampler = SubsetRandomSampler(val_list)
    test_sampler = SubsetRandomSampler(test_list)

    val_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
    val_loader = DataLoader(val_sub, batch_size=1, sampler=val_sampler, num_workers=0, pin_memory=True)

    test_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
    test_loader = DataLoader(test_sub, batch_size=1, sampler=test_sampler, num_workers=0, pin_memory=True)
    return val_loader, test_loader

#############
args = parser.parse_args()

save_best_name = "MICAD_best_" + str(args.number)

if not args.Un:
	save_best_name = "MICAD_no_Un_best"


W_name = save_best_name
experiment = wandb.init(project='Mutil_MICAD_Unet', resume='allow', anonymous='must', name=W_name)
experiment.config.update(dict(batch_size=args.batch_size, learning_rate=args.base_lr))

loss_fn = CrossEntropyLoss(ignore_index=3)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def get_current_consistency_weight(epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def val_epoch(phase, epoch, model, dataloader):
	
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	val = phase == "val"
	if val:
		model.eval()

	disc_all = []
	cup_all = []
	for data in progress_bar:

		volume_batch, label_batch = data["image"], data["mask"]
		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = model(volume_batch)
		mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()
			

		mask_true = label_batch[5]
	
		mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
		dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
		dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
		disc_all.append(dice_disc.item())
		cup_all.append(dice_cup.item())
		progress_bar.set_postfix(disc = np.mean(disc_all), cup = np.mean(cup_all))
		
	final_disc = np.mean(disc_all)
	final_cup =  np.mean(cup_all)
	mean_dice = np.mean([final_disc,final_cup])
	

	info = {"final_disc": final_disc, "final_cup":final_cup,"mean_dice":mean_dice}
	return info

#train
def sigmoid_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = input_logits
    target_softmax = target_logits
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def train_epoch(phase, epoch, model,ema_model, dataloader, loss_fn):
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	training = phase == "train"

	iter_num = 0

	if training:
		model.train()
		ema_model.train()

	for data in progress_bar:
		volume_batch, label_batch,name, ori = data["image"], data["mask"],data['name'],data['image_ori']
		volume_batch = volume_batch.to(device, dtype=torch.float32)
		if isinstance(label_batch, list):
			targets = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)
		else:
			targets = label_batch.to(device, dtype=torch.long)
		outputs = model(volume_batch)
		outputs_soft = torch.sigmoid(outputs)

		K = 4
		ema_preds = []
		for k in range(K):
			noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
			ema_inputs = volume_batch + noise
			with torch.no_grad():
				ema_pred = ema_model(ema_inputs)
			ema_preds.append(ema_pred)

		ema_preds = torch.stack(ema_preds).to(device=device)
		ema_preds_soft = torch.sigmoid(ema_preds)

		ema_uncer = -1.0 * torch.sum(ema_preds_soft * torch.log2(ema_preds_soft + 1e-6), dim=0)
		ema_preds_soft = torch.mean(ema_preds_soft, dim=0)
		threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num,
														args.max_iterations)) * np.log(2)
		mask = (ema_uncer < threshold).float()

		value2 = torch.tensor(3).to(device, dtype = torch.long)
		value_mask = torch.tensor(0).to(device, dtype = torch.float)

		binary_map = BM[args.number-3](targets)
		binary_map = binary_map.type(torch.long)

		target = torch.where(binary_map==1, targets[0], value2)
		
		# # 可视化
		# for i in range(target.shape[0]):
		# 	path = r'C:\Users\10194\Desktop\dataset\train'
		# 	label_batch0 = np.uint8(np.squeeze(np.array(targets[0].cpu())))
		# 	label_batch0[label_batch0==1] = 150
		# 	label_batch0[label_batch0==2] = 255
		# 	label_batch0[label_batch0==3] = 200
		# 	# print(path + '/mask/'+ str(iter_num)+'.png')
		# 	Image.fromarray(label_batch0).save(path + '/mask/'+ str(iter_num)+'.png')

		sup_loss = torch.mean(loss_fn(outputs, targets[0]))


		consistency_weight = get_current_consistency_weight(iter_num // 150)
		consistency_loss = torch.tensor(0)
		if epoch >= 10 and args.Un:

			con_loss = sigmoid_mse_loss(outputs_soft, ema_preds_soft)
			binary_map = binary_map.unsqueeze(dim=1)
			binary_map=torch.cat((binary_map, binary_map, binary_map), dim=1)
			mask = torch.where(binary_map==0, mask, value_mask)

			consistency_loss = torch.sum(
						mask*con_loss)/(2*torch.sum(mask)+1e-16)

		total_loss = sup_loss + consistency_weight * consistency_loss



		model.zero_grad()
		total_loss.backward()
		model.optimize()
		update_ema_variables(model, ema_model, args.ema_decay, iter_num)

		iter_num = iter_num + 1
		progress_bar.set_postfix(loss_unet=total_loss.item(), sup_loss=sup_loss.item(),consistency_loss=consistency_loss.item())
		if iter_num % 2000 == 0:
			funcy.walk(lambda model:model.update_lr(), model)
			
	mean_loss = total_loss

	outputs_image = outputs_soft.argmax(dim=1)

	info = {"loss": mean_loss,
			}
	
	experiment.log({
		"train_loss": mean_loss,
		'train_images': wandb.Image(volume_batch[0].cpu()),
		'train_masks': {
			'train_true1': wandb.Image(target[0].float().cpu()),
			'train_pred1': wandb.Image(outputs_image[0].float().cpu()),
		}
	})

	return info

#main
def main(args, device, multask=True):
	batch_size = args.batch_size
	base_lr = args.base_lr
	patience = args.patience
	def create_model(ema=False):
		model = UnFNet_singal(3, 3, device, l_rate=base_lr, pretrained=True, has_dropout=False)
		if ema:
			for param in model.parameters():
				param.detach_()  # TODO:反向传播截断
		return model
	def worker_init_fn(worker_id):
		random.seed(args.seed + worker_id)

	model = create_model()
	ema_model = create_model(ema=True)

	best_model_path = os.path.join(fs_observer, save_best_name)

	#load data
	train_sub = MyData(args.train_dataroot, DF=['BinRushed', 'MESSIDOR'], transform=True)
	train_loader = DataLoader(train_sub, batch_size=args.batch_size, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

	val_loader, test_loader = ValTest(parameters)
	total_slices = len(train_sub)


	info = {}
	epochs = range(0, args.max_iterations // total_slices + 1)
	dice=0
	epochs_since_best = 0
	for epoch in epochs:
		info["train"] = train_epoch("train", epoch, model=model, ema_model=ema_model, dataloader=train_loader,
									loss_fn=loss_fn)
		

		info["validation"] = val_epoch("val", epoch, model=model, dataloader=val_loader)

		mean_dice= info["validation"]["mean_dice"]
		if mean_dice > dice:
			torch.save(model.state_dict(),best_model_path + '.pth')
			dice = mean_dice
			epochs_since_best = 0
		else:
			epochs_since_best += 1

		if epochs_since_best > patience:  # 最大容忍不涨区间
			break
	#test
	model_test = create_model()
	model_test.load_state_dict(torch.load('MICAD_results/'+'MICAD_best_5.pth', map_location='cpu'))
	model_test.to(device)
	info = test_model(phase = "test", model = model_test, dataloader = test_loader, device = device)

	with open('MICAP_0.93.txt', 'a') as file:
		for key,v in info.items():
			file.write('{},{} '.format(key, v))
			print(key,v)
		file.write('\n')

if __name__ == '__main__':
	if not args.deterministic:
		cudnn.benchmark = True
		cudnn.deterministic = False
	else:
		cudnn.benchmark = False
		cudnn.deterministic = True

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	main(args, device, multask=True)

