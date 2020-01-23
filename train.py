import time
import sys
import tensorflow as tf
import torch.nn as nn
sys.path.append('options')
from train_options import TrainOptions
sys.path.append('data')
from data_loader import CreateDataLoader
sys.path.append('model')
from model_Loader import CreateModel
sys.path.append('util')
from utils import error as err
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
if __name__ == "__main__":
		opt = TrainOptions().parse()
		data_loader = CreateDataLoader(opt)
		model = nn.DataParallel(CreateModel(opt))

		sess = tf.Session()
		loss_g = tf.placeholder(tf.float32)
		loss_d = tf.placeholder(tf.float32)
		me1 = tf.summary.scalar('loss_g', loss_g)
		me2 = tf.summary.scalar('loss_d', loss_d)
		merged = tf.summary.merge([me1, me2])
		writer = tf.summary.FileWriter("logs", sess.graph)

		err = err(model.module.save_dir)
		for epoch in range(opt.count_epoch + 1,  opt.epochs + 1):
			epoch_start_time = time.time()
			err.initialize()

			for i, data in enumerate(data_loader):
				model.module.forward(data)
				model.module.optimize_G_parameters()
				if(i % opt.D_interval == 0):
					#print("Optimize D")
					model.module.optimize_D_parameters()

				#err.add(model.Loss_G.data.item(), model.Loss_D.data.item())
				
				err.add(model.module.Loss_G.data, model.module.Loss_D.data)

			LOSSG, LOSSD = err.print_errors(epoch)
			#summary = sess.run(merged, feed_dict={loss_g: LOSSG, loss_d: LOSSD})
			#writer.add_summary(summary, epoch)
			print('End of epoch {0} \t Time Taken: {1} sec\n'.format(epoch, time.time()-epoch_start_time))
			model.module.save_result(epoch)
			if (epoch == 1800) or (epoch=10000):
				print('Saving the model at the end of epoch {}\n'.format(epoch))
				model.save(epoch)
				strinD = '{}_net_D.path'.format(epoch)
				strinD_Path = '/content/single/{}_net_D.path'.format(epoch)
				uploaded = drive.CreateFile({'title': strinD})
				uploaded.SetContentFile(strinD_Path)
				uploaded.Upload()
				print('Uploaded file with ID {}'.format(uploaded.get('id')))
				strinG = '{}_net_G.path'.format(epoch)
				strinG_Path = '/content/single/{}_net_G.path'.format(epoch)
				uploaded = drive.CreateFile({'title': strinG})
				uploaded.SetContentFile(strinG_Path)
				uploaded.Upload()
				print('Uploaded file with ID {}'.format(uploaded.get('id')))
