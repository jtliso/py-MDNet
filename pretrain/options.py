from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = False # Change this to true if you want to use GPU

#opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
#opts['init_model_path'] = '../models/resnet18-5c106cde.pth' # PyTorch Model
#opts['init_model_path'] = '../models/imagenet-resnet-50-dag.mat' # MatLab Model, feel closer with this model
opts['init_model_path'] = 'resnet50.pth'
opts['model_path'] = '../models/mdnet_vot-otb_new2.pth'

opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 128#107
opts['padding'] = 1

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['conv','fc']
opts['lr_mult'] = {'fc':10}
opts['n_cycles'] = 50
