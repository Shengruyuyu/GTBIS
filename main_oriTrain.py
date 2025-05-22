from __future__ import absolute_import, division, print_function
import os
import torch
from utils.dataset import GraphDataset
from tensorboardX import SummaryWriter
from utils.helper import Trainer, Evaluator, collate, GradualWarmupScheduler, focal_loss
from utils.option import Options
from torch.optim import lr_scheduler
from models.GraphTransformer_1node import Classifier

args = Options().parse()
torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True
model_path = "results/saved_models/"
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = "results/run_logs/"
if not os.path.isdir(log_path): os.mkdir(log_path)

n_class = 2
train = True
test = False
graphcam = False
k = 1
log_interval_local = 6
num_epochs = 5
batch_size = 16
learning_rate = 1e-5
num_feats = 1024
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

task_name = f"GTP_Model{k}"
print(task_name)

train_set = rf"dataset/train_val_txt/train_{k}.txt"
data_path = "results/features"
val_set = rf"dataset/train_val_txt/val_{k}.txt"
save_path = f"results/evaluate_results/fold_{k}/"
os.makedirs(save_path, exist_ok=True)


###################################
print("train:", train, "test:", test, "graphcam:", graphcam)
##### Load datasets
print("preparing datasets and dataloaders......")

if test or graphcam:
    num_epochs = 1
    batch_size = 1

if train:
    resume = False
    ids_train = open(train_set).readlines()

    dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train, num_feats)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=0,
                                                   collate_fn=collate, shuffle=True, pin_memory=True, drop_last=False)
    total_train_num = len(dataloader_train) * batch_size

ids_val = open(val_set).readlines()
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val, num_feats)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=0,
                                             collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### creating models #############
print("creating models......")

model = Classifier(n_class, num_feats)
if resume:
    model.load_state_dict(torch.load(resume))
    print('loaded model{}'.format(resume))


if torch.cuda.is_available():
    model = model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)  # best:5e-4, 4e-3
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
my_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)
##################################

if not test:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(n_class, num_feats)
evaluator = Evaluator(n_class, num_feats)

bestPredEpoch = -1
best_pred = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.
    total = 0.

    current_lr = optimizer.param_groups[0]['lr']
    print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch + 1, current_lr, best_pred))

    if train:
        for i_batch, sample_batched in enumerate(dataloader_train):
            preds, labels, loss, _ = trainer.train(sample_batched, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            total += len(labels)

            trainer.metrics.update(labels, preds)

            if (i_batch + 1) % log_interval_local == 0:
                print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (
                total, total_train_num, train_loss / total, trainer.get_scores()))
                trainer.plot_cm()
        my_scheduler.step()

    if not test:
        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (
        total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
        trainer.plot_cm()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("===================================evaluating...")

            total = 0.
            batch_idx = 0

            sampleNamesList = []
            trueLabelsList = []
            predProbsList = []
            predLabelsList = []
            for i_batch, sample_batched in enumerate(dataloader_val):
                preds, labels, _, probs = evaluator.eval_test(sample_batched, model, graphcam)

                if test:
                    sampleName = sample_batched["id"][0].rsplit("_", 1)[0]
                    sampleNamesList.append(sampleName)
                    trueLabelsList.append(1 if labels == 1 else 0)
                    predProbsList.append(float(probs[0][1]))
                    predLabelsList.append(1 if preds == 1 else 0)

                total += len(labels)

                evaluator.metrics.update(labels, preds)

                if (i_batch + 1) % log_interval_local == 0:
                    print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                    evaluator.plot_cm()

            print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
            cm = evaluator.plot_cm()

            val_acc = evaluator.get_scores()
            if val_acc >= best_pred:
                best_pred = val_acc
                bestPredEpoch = epoch
                if not test:
                    print("======================================saving model...")
                    torch.save(model.state_dict(), model_path + task_name + ".pth")

            log = ""
            log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f} ----- best val = {:.4f} on {} epoch at present'.format(epoch + 1, num_epochs,
                                                                                        trainer.get_scores(),
                                                                                        evaluator.get_scores(), best_pred, bestPredEpoch + 1) + "\n"
            log += "================================\n"
            print(log)
            if test: break

            f_log.write(log)
            f_log.flush()

            writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()},
                               epoch + 1)

    trainer.reset_metrics()
    evaluator.reset_metrics()

if not test: f_log.close()
