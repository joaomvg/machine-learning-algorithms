# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd


# %%
class Tensor:
    calc_grad=True 

    """calc_grad: bool, signaling whether to carry the gradients while artihmetic operations take place"""

    def __init__(self,array,
                grad=None,
                requires_grad=False):
        """
        array: numpy array
        grad: dic={id(object): numpy.array}
        requires_grad: bool, signaling whether to calculate or not the derivative relative to this tensor
        
        """
        self.array=array
        self.requires_grad=requires_grad
        
        if requires_grad:
            name=id(self) 
            self.grad={name: self.make_grad()}
        else:
            self.grad={'none':0}
        if grad is not None:
            self.grad=grad

    @property
    def shape(self):
        return self.array.shape
    @property
    def ndim(self):
        return self.array.ndim

    @property
    def T(self):
        return self.array.T

    def transpose(self,shape):
        self.array=self.array.transpose(shape)
        if self.calc_grad:
            for w in self.grad:
                if isinstance(self.grad[w],np.ndarray):
                    self.grad[w]=self.grad[w].transpose(shape)


    def squeeze(self,axis=0):
        result=self.array.squeeze(axis)
        if self.calc_grad:
            grad={}
            for w in self.grad:
                if isinstance(self.grad[w],np.ndarray):
                    grad[w]=self.grad[w].squeeze(axis)
                else:
                    grad[w]=0
            return Tensor(result,grad=grad)
        else:
            return Tensor(result,grad='NA')
        
    def __getitem__(self,index):
        result=self.array[index]
        grad={}
        for w in self.grad:
            if isinstance(self.grad[w],np.ndarray):
                grad[w]=self.grad[w][index]
            else:
                grad[w]=0
        
        return Tensor(result,grad=grad)

    def make_grad(self,):
        shape=self.array.shape
        Kron=1
        for d in shape:
            ID=np.identity(d)
            Kron=np.tensordot(Kron,ID,axes=0)
        new_shape=[i for i in range(0,2*len(shape),2)]
        new_shape+=[i for i in range(1,2*len(shape),2)]
        Kron=Kron.transpose(new_shape)

        return Kron

    def check_grads(self,x):

        for w in self.grad:
            if w not in x.grad:
                x.grad[w]=0
        for w in x.grad:
            if w not in self.grad:
                self.grad[w]=0

    def __add__(self,x):
        
        if isinstance(x,Tensor):
            result=self.array+x.array
            if self.calc_grad:
                self.check_grads(x)
                grad={}
                for w in self.grad:
                    grad[w]=self.grad[w]+x.grad[w]
                return Tensor(result,grad=grad)
            else:
                return Tensor(result,grad='NA')

        if isinstance(x,int) or isinstance(x,float):
            result=self.array+x
            if self.calc_grad:
                return Tensor(result,grad=self.grad.copy())
            else:
                return Tensor(result,grad='NA')
    
    
    def __radd__(self,x):

        if isinstance(x,int) or isinstance(x,float):
            result=self.array+x
            if self.calc_grad:
                return Tensor(result,grad=self.grad.copy())
            else:
                return Tensor(result,grad='NA')

    def __sub__(self,x):
        
        if isinstance(x,Tensor):
            result=self.array-x.array
            if self.calc_grad:
                self.check_grads(x)
                grad={}
                for w in self.grad:
                    grad[w]=self.grad[w]-x.grad[w]

                return Tensor(result,grad=grad)
            else:
                return Tensor(result,grad='NA')

        if isinstance(x,int) or isinstance(x,float):
            result=self.array-x
            if self.calc_grad:
                return Tensor(result,grad=self.grad.copy())
            else:
                return Tensor(result,grad='NA')
    
    def __rsub__(self,x):
        
        if isinstance(x,int) or isinstance(x,float):
            result=x-self.array
            if self.calc_grad:
                grad={}
                for w in self.grad:
                    grad[w]=-self.grad[w]
                return Tensor(result,grad=grad)
            else:
                return Tensor(result,grad='NA')

    def __mul__(self,x):

        if isinstance(x,int) or isinstance(x,float):
            result=x*self.array
            if self.calc_grad:
                grad={}
                for w in self.grad:
                    grad[w]=x*self.grad[w]
                return Tensor(result,grad=grad)
            else:
                return Tensor(result,grad='NA')

        if isinstance(x,Tensor):
            result=np.tensordot(self.array,x.array,axes=([-1],[0]))
            if self.calc_grad:
                self.check_grads(x)
                grad={}
                for w in self.grad:
                    if x.grad[w] is 0:
                        grad1=0
                    else:
                        grad1=np.tensordot(self.array,x.grad[w],axes=([-1],[0]))
                        
                    if self.grad[w] is 0:
                        grad2=0
                    else:
                        i=len(self.array.shape)
                        grad2=np.tensordot(self.grad[w],x.array,axes=([i-1],[0]))
                        n1=self.grad[w].ndim
                        n2=self.array.ndim
                        n3=x.array.ndim
                        r1=[j for j in range(n2-1)]+[j for j in range(n1-1,n1+n3-2)]
                        r2=[j for j in range(n2-1,n1-1)]
                        grad2=grad2.transpose(r1+r2)
                    
                    grad[w]=grad1+grad2

                return Tensor(result,grad=grad)
            else:
                return Tensor(result,grad='NA')
    
    def __rmul__(self, x):
        if isinstance(x,int) or isinstance(x,float):
            result=x*self.array
            if self.calc_grad:
                grad={}
                for w in self.grad:
                    grad[w]=x*self.grad[w]
                return Tensor(result,grad=grad)
            else:
                return Tensor(result,grad='NA')
    
    def __neg__(self):
        result=-self.array
        if self.calc_grad:
            grad={}
            for w in self.grad:
                grad[w]=-self.grad[w]
            return Tensor(result,grad=grad)
        else:
            return Tensor(result,grad='NA')
        
    def sum(self,axis):
        result=self.array.sum(axis=axis)
        if self.calc_grad:
            grad={}
            for w in self.grad:
                if self.grad[w] is not 0:
                    grad[w]=self.grad[w].sum(axis=axis)
                else:
                    grad[w]=0
            return Tensor(result,grad=grad)
        else:
            return Tensor(result,grad='NA')

    def __repr__(self):
        return f'Tensor({self.array},dtype {self.array.dtype},requires_grad={self.requires_grad})'


# %%
class Sigmoid:
    """
    returns: Tensor with gradients
    """
    def __call__(self,x):

        u=np.exp(-x.array)
        out=1/(1+u)

        if Tensor.calc_grad:
            grad={}
            for w in x.grad:
                if x.grad[w] is not 0:
                    i=x.ndim
                    l=x.grad[w].ndim
                    expand=tuple([k for k in range(i,l)])
                    grad_func=self.grad(u)
                    grad_func=np.expand_dims(grad_func,axis=expand)
                    grad[w]=grad_func*x.grad[w]
                else:
                    grad[w]=0

            return Tensor(out,grad=grad)
        else:
            return Tensor(out,grad='NA')

    @staticmethod
    def grad(u):
        den=(1+u)*(1+u)
        gd=u/den

        return gd

class Log:

    def __call__(self,x):
        out=np.log(x.array)

        grad={}
        for w in x.grad:
            if x.grad[w] is not 0:
                i=x.ndim
                l=x.grad[w].ndim
                expand=tuple([k for k in range(i,l)])
                grad_func=self.grad(x)
                grad_func=np.expand_dims(grad_func,axis=expand)
                grad[w]=grad_func*x.grad[w]
            else:
                grad[w]=0

        return Tensor(out,grad=grad)

    @staticmethod
    def grad(x):
        gd=1/x.array

        return gd

class ReLU:
    def __call__(self,x):
        sign=(x.array<0)
        z=x.array.copy()
        z[sign]=0

        if Tensor.calc_grad:
            grad={}
            for w in x.grad:
                if x.grad[w] is not 0:
                    i=x.ndim
                    l=x.grad[w].ndim
                    expand=tuple([k for k in range(i,l)])
                    grad_func=self.grad(x,sign)
                    grad_func=np.expand_dims(grad_func,axis=expand)
                    grad[w]=grad_func*x.grad[w]
                else:
                    grad[w]=0

            return Tensor(z,grad=grad)
        else:
            return Tensor(z,grad='NA')

    @staticmethod
    def grad(x,sign):
        z=x.array.copy()
        z[sign]=0
        z[~sign]=1

        return z

class Softmax:

    def __call__(self,x):
        """calculate grads after softmaz operation

        Args:
            x (Tensor): shape=(batch,num_classes)

        Returns:
            Tensor: contains gradients relative to softmax function
        """
    
        prob=np.exp(x.array)
        Z=prob.sum(1).reshape(-1,1)
        prob=prob/Z

        if Tensor.calc_grad:
            grad={}
            for w in x.grad:
                if x.grad[w] is not 0:
                    i=x.ndim
                    l=x.grad[w].ndim
                    expand=tuple([k for k in range(i,l)])
                    grad_func=np.expand_dims(prob,axis=expand)
                    dp=grad_func*x.grad[w]
                    grad[w]=dp-grad_func*np.expand_dims(dp.sum(1),axis=1)
                else:
                    grad[w]=0

            return Tensor(prob,grad=grad)
        else:
            return Tensor(prob,grad='NA')



"""
Training models 
"""

# %%
class LinearLayer:
    def __init__(self,in_dim,out_dim,bias=True):
        self.in_dim=in_dim
        self.out_dim=out_dim

        weight_,bias_=self.init_params()

        self.weight=Tensor(weight_,requires_grad=True)
        if bias:
            self.bias=Tensor(bias_,requires_grad=True)
        else:
            self.bias=0

        self.trainable={id(self.weight): self.weight,
                        id(self.bias): self.bias}
        
    def __call__(self,x):
        """
        x: Tensor [batch,in_dim]
        """
        out=x*self.weight+self.bias
        return out

    def init_params(self):
        weight=np.random.normal(0,1,(self.in_dim,self.out_dim))
        bias=np.random.normal(0,1,(1,self.out_dim))
        return weight, bias


class FeedForward:

    def __init__(self,input_dim,hidden_dim,out_dim=1,n_hid_layers=0):
        self.train() 
        self.in_layer=LinearLayer(input_dim,hidden_dim)
        self.hid_layers=[LinearLayer(hidden_dim,hidden_dim) for i in range(n_hid_layers)]
        self.out_layer=LinearLayer(hidden_dim,out_dim)
        self.relu=ReLU()
        self.sig=Sigmoid()

    def __call__(self,x):
        """
        assume two class problem
        """
        out=self.in_layer(x)
        out=self.relu(out)
        for layer in self.hid_layers:
            out=layer(out)
            out=self.relu(out)
        out=self.out_layer(out)
        out=self.sig(out)

        return out
    
    def predict(self,x):
        """
        predict
        """
        pred=self(x)
        pred=pred.array.squeeze(1)
        y_pred=(pred.array>=0.5).astype('int8')

        return y_pred

    def train(self):
        Tensor.calc_grad=True
    
    def eval(self):
        Tensor.calc_grad=False 


class LogLoss:
    def __init__(self,model):
        self.model=model
        self.back_grads=None
        self.log=Log()

    def __call__(self,prob,y):
        
        not_y=(1-y.array).reshape(-1,1).T
        not_y=Tensor(not_y)
        y_=y.array.reshape(-1,1).T
        y_=Tensor(y_)

        not_prob=1-prob.array
        grad={}
        for w in prob.grad:
            grad[w]=-prob.grad[w]
        not_prob=Tensor(not_prob,grad=grad)

        size=1/prob.shape[0]
        L=y_*self.log(prob)+not_y*self.log(not_prob)
        L=-L.sum(axis=0)
        L=size*L

        self.back_grads=L.grad

        return L.array[0]
    
    def backward(self):
        self.model.grads=self.back_grads

class Optimizer:
    def __init__(self,model,lr=0.01):
        self.model=model
        self.lr=lr
        self.tensors=self.find_tensor()
    
    def zero_grad(self):
        for idx, tensor in self.tensors.items():
            if tensor.requires_grad:
                grad={}
                grad[idx]=tensor.grad[idx]
                tensor.grad=grad
            else:
                grad={'none':0}
                tensor.grad=grad 

    def step(self):
        if self.model.grads is not None:
            for idx, tensor in self.tensors.items():
                if idx in self.model.grads:
                    tensor.array-=self.lr*self.model.grads[idx].squeeze(0)
        else:
            print('No grads!')

    def find_tensor(self):
        tensors={}
        for _,param1 in self.model.__dict__.items():
            if isinstance(param1,Tensor):
                tensors[id(param1)]=param1
            elif hasattr(param1,'__dict__'):
                for _,param2 in param1.__dict__.items():
                    if isinstance(param2,Tensor):
                        tensors[id(param2)]=param2
        return tensors



# %%
class DataSet:
    def __init__(self,x,y,batch_size=28):
        self.data_x=x
        self.data_y=y
        self.bsz=batch_size

    def __len__(self):

        return self.data_x.shape[0]
        
    def __iter__(self):
        L=self.data_x.shape[0]
        bsz=self.bsz
        for i in range(0,L,bsz):
            batch_x=Tensor(self.data_x[i:i+bsz])
            batch_y=Tensor(self.data_y[i:i+bsz])
            yield batch_x, batch_y

# %% [markdown]
# # Training

# %%
if __name__=="__main__":
    from sklearn.datasets import load_breast_cancer
    from tqdm import tqdm 

    # %%
    data=load_breast_cancer()
    x=data['data']
    y=data['target']
    x=x/x.max()

    # %%
    data_loader=DataSet(x,y,128)
    # %%
    model=FeedForward(30,50)
    loss=LogLoss(model)
    opt=Optimizer(model,0.1)


    def train(model,loss,optimizer,data_loader,epochs):
        
        L=len(data_loader)
        model.train()
        for epoch in tqdm(range(epochs)):
            total_loss=0
            for batch in data_loader:
                x_batch, y_batch=batch
                bsz=x_batch.shape[0]

                optimizer.zero_grad()
                out=model(x_batch)
                total_loss+=loss(out,y_batch)*bsz
                loss.backward()
                opt.step()

            if epoch%10==0:
                print('Loss: ',total_loss/L)
    
    train(model,loss,opt,data_loader,20)

    # %%
    def accuracy(model,data_loader):
        acc=0
        model.eval()
        for batch in data_loader:
            x_b,y_b=batch 
            out=model(x_b)
            pred=(out.array>0.5).astype('int8')
            acc+=(y_b.array==pred.squeeze(1)).sum()
        
        return acc/x.shape[0]


    # %%
    accuracy(model,data_loader)


# %%



