import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./results/ALICC/Syn/R256')
parser.add_argument('-d1','--dir1', type=str, default='/home/tdx/ypw/simulate_data/uavid_v1.5_official_release_image/SynV3/Test/R256/GT')
parser.add_argument('-o','--out', type=str, default='./results/ALICC/Syn/R256/lpips.txt')
# parser.add_argument('-d0','--dir0', type=str, default='./results/new_ablation/net/RDCNet/R960')
# parser.add_argument('-d1','--dir1', type=str, default='/home/tdx/ypw/simulate_data/uavid_v1.5_official_release_image/SynV3/Test/R960/GT')
# parser.add_argument('-o','--out', type=str, default='./results/new_ablation/net/RDCNet/R960/lpips.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
total_dist = 0.0
num_pairs = 0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
		total_dist += dist01
		num_pairs += 1
average_dist = total_dist / num_pairs
print('Total LPIPS Distance: %.6f' % total_dist)
print('Average LPIPS Distance: %.6f' % average_dist)
f.writelines('Total LPIPS Distance: %.6f\n' % total_dist)
f.writelines('Average LPIPS Distance: %.6f\n' % average_dist)
f.close()
