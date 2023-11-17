import torch
import numpy as np
import torch
from tqdm import tqdm
from utils import dice_score
import torch.nn.functional as F
import funcy
import argparse
from networks.DD import UnFNet_singal
from torch.utils.data import DataLoader
from utils.dataset_loader_cvpr import MyData
from medpy import metric
import SimpleITK as sitk

def calculate_hd95_asd(pred, gt):

    hd = metric.binary.hd95(pred[0, ...], gt[0, ...])
    asd = metric.binary.jc(pred[0, ...], gt[0, ...])
    
    return hd, asd

def iou(matrix1, matrix2):
    intersection = np.logical_and(matrix1, matrix2)
    intersection_count = np.count_nonzero(intersection == 1)
    matrix1_count = np.count_nonzero(matrix1 == 1)
    matrix2_count = np.count_nonzero(matrix2 == 1)
    iou = intersection_count / float(matrix1_count + matrix2_count - intersection_count)
    return iou

def ave(label_batch):
	most_frequent_values, _ = torch.mode(torch.stack(label_batch), dim=0)
	return most_frequent_values


def test_model(phase, model, dataloader, device):
	
	progress_bar = tqdm(dataloader, desc="Epoch {}".format( phase))
	test = phase == "test"
	if test:
		model.eval()

	disc_dice_all = []
	cup_dice_all = []
	# disc_95hd_all = []
	# cup_95hd_all = []
	# disc_asd_all = []
	# cup_asd_all = []
	disc_iou_all =[]
	cup_iou_all =[]


	r_num = 0
	for data in progress_bar:

		volume_batch, label_batch = data["image"], data["mask"]
		volume_batch = volume_batch.to(device, dtype=torch.float32)
		label_batch = funcy.walk(lambda target: target.to(device, dtype=torch.long), label_batch)

		with torch.no_grad():
			mask_pred = model(volume_batch)
		mask_pred = F.one_hot(mask_pred.argmax(dim=1), 3).permute(0, 3, 1, 2).float().cpu()
			

		# mask_true = ave(label_batch)
		mask_true = label_batch[5]

	
		mask_true = F.one_hot(mask_true, 3).permute(0, 3, 1, 2).float().cpu()
		dice_disc = dice_score.dice_coeff(mask_pred[:, 1:2, ...], mask_true[:, 1:2, ...], reduce_batch_first=False)
		dice_cup = dice_score.dice_coeff(mask_pred[:, 2:3, ...], mask_true[:, 2:3, ...], reduce_batch_first=False)

		# hd95_disc, asd_disc = calculate_hd95_asd(np.array(mask_pred[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu')))
		# hd95_cup, asd_cup = calculate_hd95_asd(np.array(mask_pred[:, 2:3, ...].to('cpu')),np.array(mask_true[:, 2:3, ...].to('cpu')))

		iou_disc = iou(np.array(mask_pred[:, 1:2, ...].to('cpu')),np.array(mask_true[:, 1:2, ...].to('cpu')))
		iou_cup = iou(np.array(mask_pred[:, 2:3, ...].to('cpu')),np.array(mask_true[:, 2:3, ...].to('cpu')))





		disc_dice_all.append(dice_disc.item())
		cup_dice_all.append(dice_cup.item())

		# disc_95hd_all.append(hd95_disc)
		# cup_95hd_all.append(hd95_cup)

		# disc_asd_all.append(asd_disc)
		# cup_asd_all.append(asd_cup)

		disc_iou_all.append(iou_disc)
		cup_iou_all.append(iou_cup)

		r_num += 1
		progress_bar.set_postfix(disc = np.mean(disc_dice_all), cup = np.mean(cup_dice_all))
		
	dice_disc = np.mean(disc_dice_all)
	dice_cup =  np.mean(cup_dice_all)

	# hd95_disc = np.mean(disc_95hd_all)
	# hd95_cup = np.mean(cup_95hd_all)

	# asd_disc = np.mean(disc_asd_all)
	# asd_cup = np.mean(cup_asd_all)

	iou_disc = np.mean(disc_iou_all)
	iou_cup = np.mean(cup_iou_all)


	

	info = {"dice_disc": [round(dice_disc, 4), round(np.std(disc_dice_all),2)], 
	 		"dice_cup":[round(dice_cup,4),round(np.std(cup_dice_all),2)],
	 		# "95hd_disc":[round(hd95_disc,2),round(np.std(disc_95hd_all),2)],
	 		# "95hd_cup":[round(hd95_cup,2),round(np.std(cup_95hd_all),2)],
	 		# "asd_disc":[round(asd_disc,2),round(np.std(disc_asd_all),2)],
	 		# "asd_cup":[round(asd_cup,2),round(np.std(cup_asd_all),2)],
			 "iou_disc":[round(iou_disc,4),round(np.std(disc_iou_all),2)],
			 "iou_cup":[round(iou_cup,4),round(np.std(cup_iou_all),2)],

			 
			}
	return info


def create_model(device, ema=False):
	model = UnFNet_singal(3, 3, device,l_rate=0.1, pretrained=True, has_dropout=ema)
	if ema:
		for param in model.parameters():
			param.detach_()  # TODO:反向传播截断
	return model

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, default="MICAD_results/MICAD_best_6.pth", help="model_path")
	parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
	parser.add_argument("--gpu", type=int, default=0, help="gpu")
	parser.add_argument('--test_dataroot', default='./DiscRegion', type=str, help='path to test or val data')


	args = parser.parse_args()
	device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
	
	model_path = args.model_path
	model = create_model(device=device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	
	val_sub = MyData(args.test_dataroot, DF=["Magrabia"])
	val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

	info = test_model(phase = "test", model = model, dataloader = val_loader, device = device)
	print(info)
	for key,v in info.items():
		print(key,v)


