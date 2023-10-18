import torch
import copy
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# kd loss
import torch.nn.functional as F
import torch

def cal_loss(s_logits, t_logits, temperature): # s_logits: student's logits, t_logits: teacher's logits
    """
    Calculates the knowledge distillation loss between the student and teacher logits.
    
    Args:
    - s_logits (torch.Tensor): The logits of the student model.
    - t_logits (torch.Tensor): The logits of the teacher model.
    - temperature (float): The temperature used for scaling the logits.
    
    Returns:
    - loss (torch.Tensor): The knowledge distillation loss.
    """
    # Calculate the soft labels using the teacher logits and temperature
    soft_labels = F.log_softmax(t_logits / temperature, dim=-1, dtype=torch.float32)
    
    # Calculate the log probabilities using the student logits and temperature
    log_prob = F.log_softmax(s_logits / temperature, dim=-1, dtype=torch.float32)
    
    # Calculate the original Kullback-Leibler divergence loss
    ori_kld_loss = -torch.exp(soft_labels) * log_prob + torch.exp(soft_labels) * soft_labels
    
    # Calculate the mean of the sum of the Kullback-Leibler divergence loss
    loss = torch.mean(torch.sum(ori_kld_loss, dim=-1))
    
    return loss

class LGTMTeacher(object):
    def __init__(self,teacher_model,student_model,alpha_kd,alpha_kd_t,optimizer_t,scheduler_t, temperature):
        self.temperature = temperature
        self.teacher_model = teacher_model
        self.student_model = student_model
        # for student
        self.alpha_kd = alpha_kd 
        # for teacher
        self.alpha_kd_t = alpha_kd_t 
        self.optimizer_t = optimizer_t
        self.scheduler_t = scheduler_t

    def cal_stu_tea_loss(self, teacher_outputs, student_outputs, flag=1):
            """
            Calculates the student and teacher loss using the knowledge distillation technique.

            Args:
                teacher_outputs (object): The output of the teacher model.
                student_outputs (object): The output of the student model.
                flag (int): If flag=0, calculate the student loss and teacher loss simultaneously.

            Returns:
                student_loss (float): The student loss.
                teacher_loss (float): The teacher loss.
            """
            t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits # teacher's loss and logits
            loss, logits = student_outputs.loss, student_outputs.logits # student's loss and logits
            
            student_loss = None
            teacher_loss = None

            # if flag=0, calculate the student loss and teacher loss simultaneously
            if flag == 0:
                # for student
                t_soft_labels = t_logits.detach() # soft labels
                s_kld_loss = cal_loss(logits, t_soft_labels, self.temperature) # 
                student_loss = self.alpha_kd * s_kld_loss + (1- self.alpha_kd) * loss
                
            # for teacher
            soft_labels = logits.detach() 
            t_kld_loss = cal_loss(t_logits, soft_labels, self.temperature) 
            teacher_loss = self.alpha_kd_t *  t_kld_loss + (1- self.alpha_kd_t) * t_loss
        
            return student_loss, teacher_loss
        
    def step(self, inputs, eval_inputs, network_optimizer): # network_optimizer: student's opt
        self.optimizer_t.zero_grad()
        self._backward_step_unrolled(inputs, eval_inputs, network_optimizer) 
        self.optimizer_t.step()  
        self.scheduler_t.step() 

    def _backward_step_unrolled(self, inputs, eval_inputs, network_optimizer): 
        # Copy a student model and update it
        unrolled_model, dalpha = self._compute_unrolled_model(inputs, network_optimizer) # unrolled_model: 
       # student 调用 _compute_unrolled_model 方法来获取学生模型的一个副本（unrolled_model）和该模型的梯度（dalpha）。学生模型的副本用于在验证集上计算损失和梯度，以避免影响原始学生模型。

        # Sample a batch of validation set
        for k, v in eval_inputs.items():
            eval_inputs[k] = v.to(unrolled_model.device)
        unrolled_model.train()
        outputs = unrolled_model(**eval_inputs) 
        unrolled_loss = outputs[0]

        # unrolled_model: student
        unrolled_loss.backward()
        # dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()] # gradients of student's parameters on validation set

        # Calculate the Distillation Influence
        implicit_grads = self._hessian_vector_product(vector, inputs)

        eta = self.scheduler_t.get_last_lr()[0] # get the learning rate of teacher
        # update teacher here, the gradients of teacher model contains two parts
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data) # update the gradients of teacher model

        for v, g in zip(self.teacher_model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
              
    def _compute_unrolled_model(self, input, network_optimizer): # input: train batch， 用于计算学生模型的一个副本（即 "unrolled model"）和该模型的梯度。
        for k, v in input.items(): # 将输入数据移到教师模型所在的设备
            input[k] = v.to(self.teacher_model.device)
        teacher_outputs = self.teacher_model(**input) 
        student_outputs = self.student_model(**input)
        student_loss, teacher_loss = self.cal_stu_tea_loss(teacher_outputs, student_outputs, flag=0)
       
        dtheta = torch.autograd.grad(student_loss, self.student_model.parameters()) # 使用 PyTorch 的 grad 函数计算 student_loss 关于学生模型参数的梯度。
        theta = []
        index = 0

        for group in network_optimizer.param_groups: # 用于更新学生模型参数的优化器
            for p in group["params"]: # 遍历学生模型的所有参数
                # if p.grad is None:
                #     continue
                # grad = p.grad.data
                
                grad = dtheta[index] 
                index += 1
                if grad is None:
                    continue
                # grad = dtheta[index].data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = copy.deepcopy(network_optimizer.state[p]) # 用于更新学生模型参数的优化器的状态

                # State initialization
                # 这段代码是在一个循环中，该循环遍历学生模型的所有参数。对于每个参数，它模拟了优化器更新参数的过程。

                # 如果优化器的状态信息尚未初始化，它将初始化状态信息。
                # 然后，它获取状态信息和优化器的超参数，用于计算参数的更新值。
                # 最后，它更新优化步数，以记录该参数已被更新的次数。
                # 这些信息将被用于计算学生模型参数的更新值，进一步用于构建学生模型的一个副本。这个副本将被用于在验证集上计算损失和梯度，这些信息将进一步用于计算和更新教师模型的梯度。
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"]) # eps: a small constant for numerical stability
                                                             # 这行代码的结果是计算了分母项 denom，它将被用于计算参数的更新值。

                step_size = group["lr"] # 从 group 字典中获取学习率 lr，并将其赋值给 step_size
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                d = p.data.addcdiv(-step_size, exp_avg, denom) # 计算参数的更新值，
                                                               # addcdiv(-step_size, exp_avg, denom) 是一个 in-place 操作，用于将 exp_avg 除以 denom 后乘以 -step_size 并加到 p.data 上。

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    d.add_(-group["lr"] * group["weight_decay"], d) # 将 weight_decay 乘以学习率 -lr，然后乘以参数的值 d，最后加到 d 上。
                theta.append(d)
        unrolled_model = self._construct_model_from_theta(_concat(theta).data) # 用于构建学生模型的一个副本
        # _concat(theta) 是一个函数调用，它将 theta 列表中的所有张量连接成一个大张量。theta 列表包含了学生模型的每个参数的更新值。
        # _construct_model_from_theta 函数将这个大张量转换成一个学生模型的副本。
      
        # calculate the grad for teacher
        dalpha = torch.autograd.grad(teacher_loss, self.teacher_model.parameters())
            
        return unrolled_model,dalpha

    def _construct_model_from_theta(self, theta): # 是用于根据给定的参数值（在这里是梯度）构建一个新的学生模型的副本，theta（学生模型的参数梯度）
        model_new = copy.deepcopy(self.student_model) # copy a student
        model_dict = model_new.state_dict() # 获取学生模型的参数字典

        params, offset = {}, 0 # 初始化一个空字典 params 用于存储更新后的参数值，和一个偏移量 offset 用于在 theta 中定位参数值。
        for k, v in self.student_model.named_parameters():
            v_length = np.prod(v.size()) # 计算参数 v 的长度
            params[k] = theta[offset: offset + v_length].view(v.size()) #从 theta 中切片出对应的值，并调整其形状以匹配参数 v 的形状。
            offset += v_length # 更新偏移量 offset

        assert offset == len(theta) # 确保偏移量 offset 等于 theta 的长度
        model_dict.update(params) # 使用更新后的参数值 params 更新 model_dict。
        model_new.load_state_dict(model_dict) # 调用 load_state_dict() 方法将更新后的状态字典 model_dict 加载到 model_new 中。
        # return model_new.cuda()
        return model_new

    def _hessian_vector_product(self, vector, input, r=1e-2): # 用于计算Hessian矩阵与一个向量的乘积
        R = r / _concat(vector).norm() # episilon
                                        # r 是一个小的常数，用于数值稳定性。_concat(vector).norm() 计算 vector 的范数。
                                       # R 用于后续的有限差分计算。
        # vector is the gradients of the student's parameters on valuation set
        self.teacher_model.eval()
        self.student_model.eval()
        teacher_outputs = self.teacher_model(**input) # (loss), logits, (hidden_states), (attentions)

        for p, v in zip(self.student_model.parameters(), vector): # 使用 vector 更新学生模型的参数。这是为了计算Hessian向量积的一部分。
            p.data.add_(R, v) 
        student_outputs = self.student_model(**input)
        _, loss_x = self.cal_stu_tea_loss(teacher_outputs, student_outputs)
        grads_p = torch.autograd.grad(loss_x, self.teacher_model.parameters())

        for p, v in zip(self.student_model.parameters(), vector):
            p.data.sub_(2 * R, v)
        teacher_outputs = self.teacher_model(**input) # (loss), logits, (hidden_states), (attentions)
        student_outputs = self.student_model(**input)
        _, loss_y = self.cal_stu_tea_loss(teacher_outputs, student_outputs)
        grads_n = torch.autograd.grad(loss_y, self.teacher_model.parameters()) #  计算损失关于教师模型参数的梯度

        # recover the parameters of the student
        for p, v in zip(self.student_model.parameters(), vector): # 恢复学生模型的参数。
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)] # 使用有限差分方法计算Hessian向量积，并返回结果
