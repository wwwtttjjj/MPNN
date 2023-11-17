import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
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
os.environ["WANDB_MODE"] = "dryrun"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
fs_observer = os.path.join(BASE_PATH, "MPNN_results")
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
parser.add_argument('--value', type=float, default=0.92,
					help='0-1')

parser.add_argument('--number', type=int, default=6,
					help='2-6')

args = parser.parse_args()

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
			

		mask_true = label_batch[0]
	
		mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
		dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
		dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)
		disc_all.append(dice_disc.item())
		cup_all.append(dice_cup.item())
		progress_bar.set_postfix(disc = np.mean(disc_all), cup = np.mean(cup_all))
		
	final_disc = np.mean(disc_all)
	final_cup =  np.mean(cup_all)
	mean_dice = np.mean([final_disc,final_cup])
	if mean_dice > args.value:
		for data in progress_bar:
			volume_batch, label_batch, name = data["image"], data["mask"], data["name"]
			volume_batch = volume_batch.to(device, dtype=torch.float32)
			label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

			with torch.no_grad():
				mask_pred = model(volume_batch)
			mask_pred = mask_pred.argmax(dim=1)
			for i in range(mask_pred.shape[0]):
				label_batch0 = np.uint8(np.squeeze(np.array(mask_pred[i].cpu())))
				label_batch0[label_batch0==1] = 150
				label_batch0[label_batch0==2] = 255
				# print("DiscRegion/"+"Rater1/"+name[i][6:]+".tif")
				Image.fromarray(label_batch0).save("DiscRegion/"+"Rater"+str(args.number)+'/'+name[i][6:]+".tif")
	info = {"final_disc": final_disc, "final_cup":final_cup,"mean_dice":mean_dice}
	return info

#train
def sigmoid_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = input_logits
    target_softmax = target_logits
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def train_epoch(phase, epoch, model, dataloader, loss_fn):
	progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
	training = phase == "train"

	iter_num = 0

	if training:
		model.train()

	for data in progress_bar:
		volume_batch, label_batch,name, ori = data["image"], data["mask"],data['name'],data['image_ori']
		volume_batch = volume_batch.to(device, dtype=torch.float32)
		if isinstance(label_batch, list):
			targets = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)
		else:
			targets = label_batch.to(device, dtype=torch.long)
		outputs = model(volume_batch)
		sup_loss = torch.mean(loss_fn(outputs, targets[0]))
		total_loss = sup_loss
		model.zero_grad()
		total_loss.backward()
		model.optimize()

		iter_num = iter_num + 1
		progress_bar.set_postfix(loss_unet=total_loss.item())
		if iter_num % 2000 == 0:
			funcy.walk(lambda model:model.update_lr(), model)
			
	mean_loss = total_loss

	info = {"loss": mean_loss,
			}

	return info

#main
def main(args, device, multask=True):
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

	#load data
	train_sub = MyData(args.train_dataroot, DF=['BinRushed', 'MESSIDOR'], transform=True)
	train_loader = DataLoader(train_sub, batch_size=args.batch_size, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

	val_sub = MyData(args.test_dataroot, DF=['BinRushed', 'MESSIDOR'])
	val_loader = DataLoader(val_sub, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

	total_slices = len(train_sub)


	info = {}
	epochs = range(0, args.max_iterations // total_slices + 1)

	for epoch in epochs:
		info["train"] = train_epoch("train", epoch, model=model, dataloader=train_loader,
									loss_fn=loss_fn)
		

		info["validation"] = val_epoch("val", epoch, model=model, dataloader=val_loader)

		mean_dice= info["validation"]["mean_dice"]
		print(mean_dice)
		if mean_dice > args.value:
			break
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

