import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import copy
import pytorch_lightning as pl
import torch
from vilt.config import ex
from vilt.modules import ViLTransformerSS
# from vilt.datamodules.multitask_datamodule import MTDataModule
import numpy as np
from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform, pixelbert_transform_randaug
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from torch import nn
from PIL import Image
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from torch import optim as optim
from  vilt.modules import create_dataset, create_loader, test_collate_fn, train_collate_fn, mmimdb_collate_fn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
from transformers import ViltProcessor, ViltModel
import torch.nn.functional as F
from KD_loss import distillation_loss, patience_loss

# m = nn.Sigmoid()
# loss = nn.BCELoss()

patience = 0
global_acc = 0.

@ex.automain
def main(_config):
    
    
    _config = copy.deepcopy(_config)

    processor = ViltProcessor.from_pretrained("/home/kennnys/projects/model/vilt_model")

    def train(teacher_model, student_model, optimizer, dataloader, loss_fn):
        losses = []
        kl  = nn.KLDivLoss(reduction = "batchmean")
        mse = nn.MSELoss()
        print(_config["beta"])
        for batch in tqdm(dataloader):
           
            for item in batch:
                batch[item] = batch[item].cuda()

            teacher_hiddens = []
            teacher_predictions = []
           
            with torch.no_grad():
                teacher_rets = teacher_model(batch, is_mask=True, is_pkd=False)
                for ret in teacher_rets:
                    
                    teacher_hiddens.append(ret["last_hidden"])
                    teacher_predictions.append(ret["predition"])
        
            student_hiddens = []
            student_predictions = []
            student_rets = student_model(batch, is_mask=True, is_pkd=False)

            for ret in student_rets:
                student_hiddens.append(ret["last_hidden"])
                student_predictions.append(ret["predition"])



          
            



            

            loss_dl, kd_loss, ce_loss = distillation_loss(student_predictions[0],batch["labels"],teacher_predictions[0],T = _config["T"], alpha=_config["alpha"])
            
            pt_loss = _config["beta"]*patience_loss(teacher_hiddens[0], student_hiddens[0])
            loss = loss_dl + pt_loss
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            losses.append(loss.cpu().item())
        print("train Loss: ", np.array(losses).mean())


    def train_mm(model, optimizer, dataloader, loss_fn):
        losses = []
        for batch in tqdm(dataloader):
           
            for item in batch:
                batch[item] = batch[item].cuda()

            infer = model(batch)
            prediction = infer["predition"]
            loss = loss_fn(prediction,batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()  
            losses.append(loss.cpu().item())
        print("train Loss: ", np.array(losses).mean())

    
    @torch.no_grad()
    def eval_mm(model, optimizer, dataloader, _config):
        global global_acc
        model.eval()
        accuracy = 0
        all_size = 0
        y_pred = []
        y_true = []
        for batch in tqdm(dataloader):
            for item in batch:
                batch[item] = batch[item].cuda()
            infer = model(batch)
            prediction = infer["predition"]
            prediction = nn.Sigmoid()(prediction).ge(0.5)
            
            y_true.extend(batch["labels"].cpu().tolist())
            y_pred.extend(prediction.cpu().tolist())

        micro_f1 = f1_score(y_true,y_pred,average='micro')
        macro_f1 = f1_score(y_true,y_pred,average='macro')
        print("micro_f1: {}, macro_f1: {}".format(micro_f1, macro_f1))
            

        f = open(_config["teacher_dir"]+'/mmimdb_'+str(_config["num_layers"])+'_layer.txt', "a+")
        f.writelines("micro_f1: {}, macro_f1: {}".format(micro_f1, macro_f1) + "\n")
        f.close()

            
        if micro_f1 > global_acc:
            global_acc = micro_f1
            torch.save(model, _config["teacher_dir"]+'/mmimdb_'+str(_config["num_layers"])+'_layer.pth')

    @torch.no_grad()
    def eval(model, optimizer, dataloader, _config):
        global patience
        global global_acc

        model.eval()
        accuracy = 0
        all_size = 0
        y_pred = []
        y_true = []
        for batch in tqdm(dataloader):
            for item in batch:
                batch[item] = batch[item].cuda()
            infer = model(batch)[0]
            prediction = infer["predition"]
            _, prediction = prediction.max(1)
            accuracy += (batch["labels"]==prediction).sum() 
            y_true.extend(batch["labels"].cpu().tolist())
            y_pred.extend(prediction.cpu().tolist())
            all_size += batch["labels"].size(0)
        if _config["task_name"] == "hf":
            print("acc: ", accuracy/all_size)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            print("acc:{}, prec:{}, rec:{}, f1:{}, auc:{}".format(acc , prec, rec, f1, auc))

            f = open(_config["pkd_dir"]+'/hf_'+str(_config["num_layers"])+'_layer_our.txt', "a+")
            f.writelines("acc:{}, prec:{}, rec:{}, f1:{}, auc:{}".format(acc , prec, rec, f1, auc) + "\n")
            f.close()

            
            if acc > global_acc:
                global_acc = acc
                torch.save(model.state_dict(), _config["pkd_dir"]+'/hf_'+str(_config["num_layers"])+'_layer_our.pth')



        if _config["task_name"] == "food101":
            acc = accuracy_score(y_true, y_pred)
            print("acc: ", acc)

       
            f = open(_config["pkd_dir"]+'/food101_'+str(_config["num_layers"])+'_layer_our.txt', "a+")
            f.writelines("acc :" + str(acc) + "\n")
            f.close()
            
            if acc > global_acc:
                global_acc = acc
                torch.save(model.state_dict(), _config["pkd_dir"]+'/food_101_'+str(_config["num_layers"])+'_layer_our.pth')

        if _config["task_name"] == "dom":
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            print("acc:{}, prec:{}, rec:{}, f1:{}, auc:{}".format(acc , prec, rec, f1, auc))

            f = open(_config["teacher_dir"]+'/dom_'+str(_config["num_layers"])+'_layer_our.txt', "a+")
            f.writelines("acc:{}, prec:{}, rec:{}, f1:{}, auc:{}".format(acc , prec, rec, f1, auc) + "\n")
            f.close()

            if acc > global_acc:
                global_acc = acc
                torch.save(model.state_dict(), _config["teacher_dir"]+'/dom_'+str(_config["num_layers"])+'_layer_our.pth')

        if _config["task_name"] == "ve":
            acc = accuracy_score(y_true, y_pred)
            print("acc: ", acc)

       
            f = open(_config["pkd_dir"]+'/ve_mage_'+str(_config["num_layers"])+'_layer_our.txt', "a+")
            f.writelines("acc :" + str(acc) + "\n")
            f.close()
            
            if acc > global_acc:
                global_acc = acc
                torch.save(model.state_dict(), _config["pkd_dir"]+'/ve_mage_'+str(_config["num_layers"])+'_layer_our.pth')
            else:
                patience += 1

        model.train()

        


    def train_hf(_config, studen_deep):

        global global_acc

        global_acc =0.

        _config["class_number"] = 2


        _config["task_name"] ="hf"
        
        loss_fn=torch.nn.CrossEntropyLoss()


        _config["num_layers"] = 12 # teacher layer deep
        teacher_model = ViLTransformerSS(_config).cuda()
        teacher_model.load_state_dict(torch.load("/home/kennnys/projects/sp_vilt/teachers/hf_12_layer.pth"))


        _config["num_layers"] = studen_deep
        student_model = ViLTransformerSS(_config).cuda()
        student_data = torch.load("/home/kennnys/projects/sp_vilt/teachers/hf_"+str(studen_deep)+"_layer.pth")
        student_model.load_state_dict(student_data)

        # teacher_model = torch.load("/home/liu/ViLT/teachers/hf_12_layer.pth")
        # teacher_model = ViLTransformerSS(_config).cuda()
        
        # teacher_model.load_state_dict(torch.load("/home/liu/ViLT/teachers/hf_12_layer.pth"))

        datasets = create_dataset("hf", _config)

        samplers = [None, None, None]

        train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                            batch_size=[_config['batch_size']]+[_config['batch_size']]*2,
                                                            num_workers=[8,8,8],is_trains=[True,False,False], 
                                                            collate_fns=[test_collate_fn,test_collate_fn,test_collate_fn])
        def create_optimizer(model):
            parameters = model.parameters()
            opt_args = dict(lr=2e-5,weight_decay=0.001)
            optimizer = optim.Adam(parameters, **opt_args)
            return optimizer

        optimizer = create_optimizer(student_model)

        for epoch in range(_config["max_epoch"]):
            train(teacher_model, student_model, optimizer, train_loader, loss_fn)
            eval(student_model, optimizer, test_loader, _config)


    def train_mmimdb(_config):

        global global_acc

        global_acc = 0.

        _config["class_number"] = 24

        _config["task_name"] ="mmimdb"
        
        loss_fn=nn.MultiLabelSoftMarginLoss()

        model = ViLTransformerSS(_config).cuda()

        datasets = create_dataset("mmimdb", _config)

        samplers = [None, None, None]

        train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                            batch_size=[_config['batch_size']]+[_config['batch_size']]*2,
                                                            num_workers=[8,8,8],is_trains=[True,False,False], 
                                                            collate_fns=[mmimdb_collate_fn,mmimdb_collate_fn,mmimdb_collate_fn])
        def create_optimizer(model):
            parameters = model.parameters()
            opt_args = dict(lr=1e-4,weight_decay=0.02)
            optimizer = optim.AdamW(parameters, **opt_args)
            return optimizer

        optimizer = create_optimizer(model)

        for epoch in range(_config["max_epoch"]):
            train_mm(model, optimizer, train_loader, loss_fn)
            eval_mm(model, optimizer, test_loader, _config)

    def train_food(_config, studen_deep):


        global global_acc

        global_acc = 0.
        _config["task_name"] ="food101"
        _config["class_number"] = 101
        loss_fn=torch.nn.CrossEntropyLoss()


        _config["num_layers"] = 12 # teacher layer deep
        teacher_model = ViLTransformerSS(_config).cuda()
        teacher_model.load_state_dict(torch.load("/home/kennnys/projects/sp_vilt/teachers/food_101_12_layer.pth"))
        # teacher_model.load("/home/x320/sp_vilt/outputs/food_101_12_layer.pth")

        _config["num_layers"] = studen_deep
        student_model = ViLTransformerSS(_config).cuda()
        student_data = torch.load("/home/kennnys/projects/sp_vilt/teachers/food_101_"+str(studen_deep)+"_layer.pth")
        student_model.load_state_dict(student_data)
        
        
        datasets = create_dataset("food101", _config)
        samplers = [None, None, None]
        train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                            batch_size=[_config['batch_size']]+[_config['batch_size']]*2,
                                                            num_workers=[8,8,8],is_trains=[True,False,False], 
                                                            collate_fns=[train_collate_fn,train_collate_fn,train_collate_fn])
        def create_optimizer(model):
            parameters = model.parameters()
            opt_args = dict(lr=2e-5,weight_decay=0.01)
            optimizer = optim.AdamW(parameters, **opt_args)
            return optimizer

        optimizer = create_optimizer(student_model)
        eval(student_model, optimizer, test_loader, _config)
        for epoch in range(_config["max_epoch"]):
            train(teacher_model, student_model, optimizer, train_loader, loss_fn)
            eval(student_model, optimizer, test_loader, _config)

    def train_ve(_config,studen_deep):


        global global_acc
        global patience

        global_acc = 0.
        _config["task_name"] ="ve"
        _config["class_number"] = 3
        loss_fn=torch.nn.CrossEntropyLoss()
        #model = ViLTransformerSS(_config).cuda()



        _config["num_layers"] = 12 # teacher layer deep
        teacher_model = ViLTransformerSS(_config).cuda()
        teacher_model.load_state_dict(torch.load("/home/kennnys/projects/sp_vilt/teachers/ve_12_layer.pth"))

        _config["num_layers"] = studen_deep
        student_model = ViLTransformerSS(_config).cuda()
        student_data = torch.load("/home/kennnys/projects/sp_vilt/teachers/ve_" + str(studen_deep)+"_layer.pth")
        student_model.load_state_dict(student_data)

        print(_config["max_epoch"])
        
        
        

        
        datasets = create_dataset("ve", _config)
        samplers = [None, None, None]
        train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                            batch_size=[_config['batch_size']]+[_config['batch_size']]*2,
                                                            num_workers=[8,8,8],is_trains=[True,False,False], 
                                                            collate_fns=[train_collate_fn,train_collate_fn,train_collate_fn])
        def create_optimizer(model):
            parameters = model.parameters()
            opt_args = dict(lr=2e-5,weight_decay=0.01)
            optimizer = optim.AdamW(parameters, **opt_args)
            return optimizer

        optimizer = create_optimizer(student_model)
        for epoch in range(_config["max_epoch"]):
            if patience > 5:
                break
            train(teacher_model, student_model, optimizer, train_loader, loss_fn)
            eval(student_model, optimizer, test_loader, _config)

    def train_dom(_config):


        global global_acc

        global_acc = 0.
        _config["task_name"] ="dom"
        _config["class_number"] = 2
        _config["max_epoch"] = 5
        loss_fn=torch.nn.CrossEntropyLoss()
        model = ViLTransformerSS(_config).cuda()

        

        
        datasets = create_dataset("dom", _config)
        samplers = [None, None, None]
        train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                            batch_size=[_config['batch_size']]+[_config['batch_size']]*2,
                                                            num_workers=[8,8,8],is_trains=[True,False,False], 
                                                            collate_fns=[train_collate_fn,train_collate_fn,train_collate_fn])
        def create_optimizer(model):
            parameters = model.parameters()
            opt_args = dict(lr=2e-5,weight_decay=0.01)
            optimizer = optim.AdamW(parameters, **opt_args)
            return optimizer

        optimizer = create_optimizer(model)
        for epoch in range(_config["max_epoch"]):
            train(model, optimizer, train_loader, loss_fn)
            eval(model, optimizer, test_loader, _config)
           

    for i in [3]:
        _config["num_layers"] = i
        train_hf(_config, i)


        

        


    



    
    

    

    
    