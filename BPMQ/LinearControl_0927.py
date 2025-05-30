import numpy as np
from copy import deepcopy as copy
from scipy import optimize
import matplotlib.pyplot as plt

class LinearModel:
    def __init__(self, input_dim, output_dim, model_accuracy_tolerance=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_accuracy_tolerance = model_accuracy_tolerance
    
    def __call__(self, x):
        return self.unnormalize_output(np.matmul(self.normalize_input(x), self.w.T) + self.b)
    
    def _get_loss_ftn(self, x, y, yerr=None, weights_each_sample=None, emphasis_recent_data=True):
        if self.model_accuracy_tolerance is None:
            weights_each_dim = np.ones(self.output_dim)
        else:
            weights_each_dim = (self.ymax - self.ymin) / (self.model_accuracy_tolerance)
        
        if weights_each_sample is None:
            weights_each_sample = np.ones(len(x))
        
        if emphasis_recent_data and len(weights_each_sample)>self.output_dim+1:
            weights_each_sample[-self.output_dim + 1:] *= 2
        
        def loss_ftn(theta):
            nw = self.input_dim * self.output_dim
            w = theta[  :nw].reshape(self.output_dim, self.input_dim)
            b = theta[nw:nw+self.output_dim].reshape(1, self.output_dim)
            y_pred = np.matmul(x, w.T) + b
            losses =  (y - y_pred)
            if yerr is not None:
                losses = losses/((yerr+1e-6)/(yerr.mean()+1e-6))
            losses = (losses * weights_each_dim[None, :]) ** 2
            losses *= weights_each_sample[:, None]
            return losses.mean()
        
        return loss_ftn
    
    def train(self, x, y, yerr=None, weights_each_sample=None, num_restarts=20, emphasis_recent_data=True, **scipy_optimize_kwargs):
        self.xmin = np.min(x, axis=0, keepdims=True)
        self.xmax = np.max(x, axis=0, keepdims=True)
        self.ymin = np.min(y, axis=0, keepdims=True)
        self.ymax = np.max(y, axis=0, keepdims=True)
        
        x = self.normalize_input(x)
        y = self.normalize_output(y)
        if yerr is not None:
            yerr = yerr/(self.ymax - self.ymin)
        
        loss_ftn = self._get_loss_ftn(x, y, yerr=yerr, weights_each_sample=weights_each_sample, emphasis_recent_data=emphasis_recent_data)
        best_loss = np.inf
        best_theta = None
        
        for i in range(num_restarts):
            if i == 0 and hasattr(self, 'theta'):
                theta0 = self.theta
            else:
                theta0 = np.random.randn((self.input_dim + 1) * self.output_dim)
            
            result = optimize.minimize(loss_ftn, theta0, **scipy_optimize_kwargs)
            
            if hasattr(result, 'fun'):
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_theta = result.x
        
        if best_theta is None:
            raise ValueError("Optimization did not converge.")
        
        self.theta = best_theta
        nw = self.input_dim * self.output_dim
        self.w = best_theta[:nw].reshape(self.output_dim, self.input_dim)
        self.b = best_theta[nw:nw + self.output_dim].reshape(1, self.output_dim)
    
    def normalize_input(self, x):
        return (x - self.xmin) / (self.xmax - self.xmin)
    
    def normalize_output(self, y):
        return (y - self.ymin) / (self.ymax - self.ymin)
    
    def unnormalize_input(self, x):
        return x * (self.xmax - self.xmin) + self.xmin
    
    def unnormalize_output(self, y):
        return y * (self.ymax - self.ymin) + self.ymin


class LinearControl:
    def __init__(self,
                 x0, dx, xmin, xmax,
                 goal, goal_tol, 
                 evaluator,
                 input_RDs,
                 output_RDs,
                 ):
        
        self.x0 = np.array(x0)
        self.dx = np.array(dx)
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        
        self.goal = np.array(goal)
        self.goal_tol = np.array(goal_tol)
        
        self.evaluator  = evaluator
        self.input_RDs  = input_RDs
        self.output_RDs = output_RDs
        self.l_evaluated_data = []
        
        
    def train_model(self, weights_each_sample=None, num_restarts=20, emphasis_recent_data=True, **scipy_optimize_kwargs):
        if not hasattr(self, 'model'):
            self.model = LinearModel(self.train_X.shape[1], self.train_Y.shape[1])
        
        self.model.train(
            self.train_X,
            self.train_Y,
            yerr = self.train_Yerr,
            weights_each_sample=weights_each_sample,
            num_restarts=num_restarts,
            emphasis_recent_data=emphasis_recent_data,
            **scipy_optimize_kwargs,
        )
    
    def _evaluate(self,x):
        df = self.evaluator(x)
        self.l_evaluated_data.append(df)
        x = df[self.input_RDs].mean().values
        y = df[self.output_RDs]
        y_err = y.std().values
        y = y.mean().values
        return x,y,yerr
    
    def initialize(self):
        x0 = self.x0
        dx = self.dx
        x_,y_,yerr_ = self._evaluate(x0)
        x = [x_]
        y = [y_]
        yerr = [yerr_]
        for i, dx_ in enumerate(dx):
            x_ = copy(x0)
            x_[i] += dx_
            x_,y_,yerr_ = self._evaluate(x_)
            x.append(x_)
            y.append(y_)
            yerr.append(yerr_)
        self.train_X = np.array(x)
        self.train_Y = np.array(y)
        self.train_Yerr = np.array(yerr)
        self.train_model()
                
    def get_loss_ftn(self):        
        def loss_ftn(x):
            x = x*(self.xmax-self.xmin) + self.xmin
            pred_y = self.model(x.reshape(1,-1)).reshape(-1)
            loss = np.mean(((pred_y - self.goal)/self.goal_tol)**2)
            return loss
        return loss_ftn

    def query_candidate(self,num_restarts=20,verbose=False,**scipy_optimize_kwargs):
        loss_ftn = self.get_loss_ftn()
        best_loss = np.inf
        best_sol  = None
        for i in range(num_restarts):
            initial_guess = np.random.rand(len(self.xmin))  # Random initialization in normalized space
            result = optimize.minimize(loss_ftn, initial_guess, bounds=[(0,1)]*len(self.xmin), **scipy_optimize_kwargs)
            
            if hasattr(result,'fun'):
                if best_loss > result.fun:
                    best_loss = result.fun
                    best_sol = result.x*(self.xmax-self.xmin) + self.xmin
                    
            if verbose:
                print(f'{i}-th restart in query_candidate. Best loss: {best_loss}, Best solution: {best_sol}')

        return best_sol
    
    def iterate(self,verbose=False):
        candidates = self.query_candidate()
        x_, y_, yerr_ = self._evaluate(candidates)
        self.train_X = np.vstack((self.train_X, [x_]))
        self.train_Y = np.vstack((self.train_Y, [y_]))
        self.train_Yerr = np.vstack((self.train_Yerr, [yerr_]))
        self.train_model()
        return x_, y_
        
    def run(self,budget):
        if not hasattr(self, "train_X"):
            self.initialize()
            budget = budget - len(self.train_X)
        for i in range(budget):
            x, y = self.iterate()

     
            
class LinearControl_virtual_evaluator:
    def __init__(self,input_CSETs,  output_RDs,
                      input_bounds=None, output_bounds=None):
        self.input_dim = len(input_CSETs)
        self.output_dim = len(output_RDs)
        self.input_CSETs = input_CSETs
        self.output_RDs  = output_RDs
        if input_bounds is None:
            self.train_X = np.random.randn(input_dim+1, input_dim)
        else:
            input_bounds = np.array(input_bounds)
            self.train_X = np.random.rand(input_dim+1, input_dim)*(input_bounds[:,1]-input_bounds[:,0]) + input_bounds[:,0]
        if output_bounds is None:
            self.train_Y = np.random.randn(input_dim+1,output_dim)
        else:
            output_bounds = np.array(output_bounds)
            self.train_Y = np.random.rand(input_dim+1, output_dim)*(output_bounds[:,1]-output_bounds[:,0]) + output_bounds[:,0]
        
        self.model = LinearModel(input_dim,output_dim)
        self.model.train(self.train_X,self.train_Y)
    def __call__(self,x):
        
        # To Do 
        return x, self.model(x.reshape(1,-1)).reshape(-1), np.ones(self.output_dim)*1e-6
        
  
        
# class virtual_evaluator:
    # def __init__(self,
                 # input_CSETs: List[str],
                 # input_RDs  : List[str],
                 # output_RDs : List[str],
                 # **kws,
                 # ):    
        # assert isinstance(input_CSETs, list), f"Expected input_CSETs to be of type list, but got {type(input_CSETs).__name__}"
        # assert isinstance(input_RDs  , list), f"Expected input_RDs to be of type list, but got {type(input_RDs).__name__}"
        # assert isinstance(output_RDs , list), f"Expected output_RDs to be of type list, but got {type(output_RDs).__name__}"
        # self.input_CSETs = input_CSETs
        # self.input_RDs   = input_RDs
        # self.output_RDs  = output_RDs
        # self.input_dim  = len(input_CSETs)
        # self.output_dim = len(output_RDs)
        
        # self.train_X = np.random.randn(self.input_dim+1,self.input_dim)
        # self.train_Y = np.random.randn(self.input_dim+1,self.output_dim)
        # self.model = LinearModel(self.input_dim, self.output_dim)
        # self.model.train(self.train_X, self.train_Y)

    # def __call__(self,x):
        # y = self.model(x.reshape(1,-1)).reshape(-1) 
        # data = df.DataFrame(np.concatenate((x,x,y)).reshape(1,-1), columns = self.input_CSETs+self.input_RDs+self.output_RDs)
        # return data, None
        
    # def submit(self, x, 
        # ):
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # future = executor.submit(self.__call__, x)
        # return future
        
    # def get_result(self, future: concurrent.futures.Future):
        # try:
            # data, ramping_data = future.result()
            # return data, ramping_data
        # except Exception as e:
            # raise RuntimeError(f"Error retrieving results: {e}")