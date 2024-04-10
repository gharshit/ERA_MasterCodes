from DataLoader import *
from Models import *
from Train import *
from Utils import *

class A10():
    def __init__(self):
        self.batchsize = 512
        self.lr = 0.01
        self.weightdecay = 0.0001        
        self.losscriteria = "crossentropy"
        self.num_epochs = 24
        self.optimizeralgo = "ADAM"
        self.dropoutnum = 0.07
        self.numworkers = 4
        self.onecyclestatus = True
        self.SEED = 4
        self.modelclass = Model_CustomResNet
        self.loader = get_CIFARdataset_with_loader
        self.trainfunc = train_ocp
        self.testfunc =  test
        self.device = setdevice()                          # set gpu if available
        self.pctabs = 5
        self.divfactor = 10
        self.finaldivfactor = 1
        
        # variables to store data to plot accuracy and loss graphs
        self.trainlosses = []
        self.testlosses = []
        self.trainacc = []
        self.testacc = []
        self.epochlr = []


    def loadmydata(self):
        ## Get CIFAR dataset and pass it to loader
         _ ,self.test_loader                = self.loader('test',self.loaderconfig())   # load test data
        self.label_, self.train_loader      = self.loader('train',self.loaderconfig())  # load train data


    def loadmymodel(self):
        self.model =  self.modelclass(dropout_value = self.dropoutnum).to(self.device)

    def giveoptimizer(self):
        # Set up the optimizer as ADAM with learning rate and weight decay as 0
        self.optimizer =  get_optimizer(self.model,self.optimizeralgo, self.lr, self.weightdecay)

    def giveloss(self):
        self.loss =  get_loss(self.losscriteria)

    def LRfinder(self):
        # Call LR finder to get MAX_LR for OCP
        lr_finder = initiateLRfinder(self.train_loader, self.model, self.optimizer, self.loss, self.device)
        self.suggested_lr = lr_finder.history["lr"][lr_finder.history["loss"].index(lr_finder.best_loss)]
        lr_finder.reset() # to reset the model and optimizer to their initial state


    def scheduleOCP(self):
        self.ocp_parameters = {'max_lr':self.suggested_lr,
        'epochs': self.num_epochs,
        'steps_per_epoch':len(self.train_loader),
        'pct_start': self.pctabs/self.num_epochs,
        'div_factor':self.divfactor,
        'final_div_factor':self.finaldivfactor
        }
        self.scheduler =  get_scheduler(self.optimizer, onecycle = self.onecyclestatus, **self.ocp_parameters)


    def loaderconfig(self):
        # Create a dictionary of keyword arguments (kwargs) for DataLoader
        kwargs = {
            'batch_size': self.batchsize,    # Set the batch size for each batch of data
            'shuffle': True,                 # ensures that the model encounters a diverse mix of data during training, leading to better learning and generalization (during testing, the order of data doesn’t impact the model’s predictions)
            'num_workers': self.numworkers,  # Number of worker threads for data loading ( each worker will deal with batchsize/num_workers set of data under a batch) # parallel processing-> reduces overall time
            'pin_memory': True               # Enable pinning memory for faster GPU transfer
        }
        return kwargs


    def runmymodel(self):
        for epoch in range(1, self.num_epochs+1):
            # Print the current epoch number
            _ = epochLR(epoch, self.scheduler,self.epochlr)

            # Call the train function, passing in the model, device, data loader, scheduler, optimizer, and loss function
            self.trainfunc(self.model, self.device, self.train_loader, self.optimizer, self.loss, self.scheduler, self.trainlosses, self.trainacc)

            # After training, evaluate the model on the same training data loader (should be test_loader for evaluation)
            self.missclassifedimages = self.test(self.model, self.device, self.test_loader, self.loss, self.testlosses, self.trainacc)


    


if __name__ == '__main__':
    
    a10_hyp = A10()      # initialize hyperparameters
    print("Hyperparamaters initialzed...")

    # set seed
    ####### Set Seed ##########
    torch.manual_seed(a10_hyp.SEED)
    if a10_hyp.device == 'cuda':
        torch.cuda.manual_seed(a10_hyp.SEED)

    
    # a10_hyp.loadmydata()
    # print("CIFAR10 dataset loaded...")

    # # label_map = {v: k for k, v in label_.items()}                             # get label map
    # # print("Target label map: ",label_map)


    
    # # Initialize the model and move it to the device ( GPU )
    # a10_hyp.loadmymodel()
    # print("Model Initialized...")


    # # Set up the optimizer
    # a10_hyp.giveoptimizer()

    # # Get the loss function 
    # a10_hyp.giveloss()

    # # run LR finder
    # a10_hyp.LRfinder()

    # # Initiate OCP
    # a10_hyp.scheduleOCP()


    # #run the model
    # a10_hyp.runmymodel()







