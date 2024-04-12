from DataLoader import *
from Models import *
from Train import *
from Utils import *
from torchsummary import summary
import warnings

#Ignore warning for now
warnings.filterwarnings('ignore')
import multiprocessing as mp

class A10():
    def __init__(self):
        # Initializing various hyperparameters for the model training
        self.batchsize = 512             # Batch size for the data loader
        self.momentum = None             # Momentum (regularization term) for the optimizer
        self.lr = 0.01                   # Learning rate for the optimizer
        self.weightdecay = 0.0001        # Weight decay (regularization term) for the optimizer
        self.losscriteria = "crossentropy"  # Loss function criteria
        self.num_epochs = 24             # Number of training epochs
        self.optimizeralgo = "ADAM"      # Optimizer algorithm
        self.dropoutnum = 0.07           # Dropout rate for the model (to prevent overfitting)
        self.numworkers = 4              # Number of workers for data loading
        self.onecyclestatus = True       # Whether to use One Cycle Policy for learning rate scheduling
        self.SEED = 4                    # Seed for random number generators for reproducibility
        self.modelclass = Model_CustomResNet  # Model class to be used
        self.trainfunc = train_ocp       # Training function
        self.testfunc = test             # Testing function
        self.device = setdevice()        # Set device to GPU if available, else CPU
        self.pctabs = 5                  # Percentage of total steps where the learning rate increases in One Cycle Policy
        self.divfactor = 10              # Initial division factor for the learning rate in One Cycle Policy
        self.finaldivfactor = 1          # Final division factor for learning rate in One Cycle Policy
        self.meanlist = mean_list        # mean of dataset
        self.stdlist = std_list          # std of dataset
        self.imageinputsize = (3,32,32)  # set image input size
        
        # Variables to store data for plotting accuracy and loss graphs
        self.trainlosses = []
        self.testlosses = []
        self.trainacc = []
        self.testacc = []
        self.epochlr = []

    def loaderconfig(self):
        # Create a dictionary of DataLoader configuration options
        kwargs = {
            'batch_size': self.batchsize,   # Set the batch size
            'shuffle': True,                # Shuffle data to ensure diverse batches
            'num_workers': self.numworkers, # Number of worker threads for data loading
            'pin_memory': True              # Enable pinning memory for faster GPU transfers
        }
        return kwargs
        
    def loadmydata(self):
        # Load dataset and its loader
        # Load test data
        _, self.test_loader = get_CIFARdataset_with_loader('test', self.loaderconfig())
        # Load train data
        self.labels, self.train_loader = get_CIFARdataset_with_loader('train', self.loaderconfig())

        print("Dataset loaded...")

    def loadmymodel(self):
        # Load the model and move it to the configured device
        self.model = self.modelclass(dropout_value=self.dropoutnum).to(self.device)
        print("Model Initialized. Below is the summary...\n")
        summary(self.model, input_size=self.imageinputsize)
        

    def giveoptimizer(self):
        # Set up the optimizer with specified algorithm, learning rate, and weight decay
        self.optimizer = get_optimizer(self.model, self.optimizeralgo, self.lr, self.weightdecay,self.momentum)

    def giveloss(self):
        # Define the loss function based on the specified criteria
        self.loss = get_loss(self.losscriteria)

    def LRfinder(self):
        # Use a Learning Rate Finder to determine the optimal maximum learning rate
        self.lr_finder = initiateLRfinder(self.train_loader, self.model, self.optimizer, self.loss, self.device)
        self.lr_finder.reset()  # Reset the model and optimizer to their initial state

    def scheduleOCP(self,maxlr):
        self.suggested_lr = maxlr    # this is taken from LRfinder suggested LR
        # Configure the One Cycle Policy parameters and scheduler
        self.ocp_parameters = {
            'max_lr': self.suggested_lr,
            'epochs': self.num_epochs,
            'steps_per_epoch': len(self.train_loader),
            'pct_start': self.pctabs / self.num_epochs,
            'div_factor': self.divfactor,
            'final_div_factor': self.finaldivfactor
        }
        self.scheduler = get_scheduler(self.optimizer, onecycle=self.onecyclestatus, **self.ocp_parameters)


    def runmymodel(self):
        for epoch in range(1, self.num_epochs + 1):
            # Iterate over each epoch
            _ = epochLR(epoch, self.scheduler, self.epochlr)  # Update learning rate for the epoch

            # Call the training function
            self.trainfunc(self.model, self.device, self.train_loader, self.optimizer, self.loss, self.scheduler, self.trainlosses, self.trainacc)

            # Evaluate the model on the test data loader
            self.missclassifiedimages = self.testfunc(self.model, self.device, self.test_loader, self.loss, self.testlosses, self.testacc)

    def modelplots(self):
        post_accuracyplots(self.trainlosses,self.testlosses,self.trainacc,self.testacc)


    def missclassplots(self):
        showmisclassifiedsamples(self.missclassifiedimages,{v: k for k, v in self.labels.items()},"Session 11",self.meanlist,self.stdlist)

        




class A11():
    def __init__(self):
        # Initializing various hyperparameters for the model training
        self.batchsize = 512             # Batch size for the data loader
        self.lr = 0.01                   # Learning rate for the optimizer
        self.momentum = None             # Momentum (regularization term) for the optimizer
        self.weightdecay = 0.0001        # Weight decay (regularization term) for the optimizer
        self.losscriteria = "crossentropy"  # Loss function criteria
        self.num_epochs = 20             # Number of training epochs
        self.optimizeralgo = "ADAM"      # Optimizer algorithm
        self.dropoutnum = 0.07           # Dropout rate for the model (to prevent overfitting)
        self.numworkers = 4              # Number of workers for data loading
        self.onecyclestatus = True       # Whether to use One Cycle Policy for learning rate scheduling
        self.SEED = 4                    # Seed for random number generators for reproducibility
        self.modelclass = ResNet18()  # Model class to be used
        self.trainfunc = train_ocp       # Training function
        self.testfunc = test             # Testing function
        self.device = setdevice()        # Set device to GPU if available, else CPU
        self.pctabs = 5                  # Percentage of total steps where the learning rate increases in One Cycle Policy
        self.divfactor = 100             # Initial division factor for the learning rate in One Cycle Policy ; Determines the initial learning rate via initial_lr = max_lr/div_factor
        self.finaldivfactor = 1          # Final division factor for learning rate in One Cycle Policy ; Determines the minimum learning rate via min_lr = initial_lr/final_div_factor
        self.meanlist = mean_list        # mean of dataset
        self.stdlist = std_list          # std of dataset
        self.imageinputsize = (3,32,32)  # set image input size
        
        # Variables to store data for plotting accuracy and loss graphs
        self.trainlosses = []
        self.testlosses = []
        self.trainacc = []
        self.testacc = []
        self.epochlr = []

    def loaderconfig(self):
        # Create a dictionary of DataLoader configuration options
        kwargs = {
            'batch_size': self.batchsize,   # Set the batch size
            'shuffle': True,                # Shuffle data to ensure diverse batches
            'num_workers': self.numworkers, # Number of worker threads for data loading
            'pin_memory': True              # Enable pinning memory for faster GPU transfers
        }
        return kwargs
        
    def loadmydata(self):
        # Load dataset and its loader
        # Load test data
        _, self.test_loader = get_CIFARdataset_with_loader('test', self.loaderconfig())
        # Load train data
        self.labels, self.train_loader = get_CIFARdataset_with_loader('train', self.loaderconfig())

        print("Dataset loaded...")

    def loadmymodel(self):
        # Load the model and move it to the configured device
        self.model = self.modelclass.to(self.device)
        print("Model Initialized. Below is the summary...\n")
        summary(self.model, input_size=self.imageinputsize)
        

    def giveoptimizer(self):
        # Set up the optimizer with specified algorithm, learning rate, and weight decay
        self.optimizer = get_optimizer(self.model, self.optimizeralgo, self.lr, self.weightdecay ,self.momentum)

    def giveloss(self):
        # Define the loss function based on the specified criteria
        self.loss = get_loss(self.losscriteria)

    def LRfinder(self):
        # Use a Learning Rate Finder to determine the optimal maximum learning rate
        self.lr_finder = initiateLRfinder(self.train_loader, self.model, self.optimizer, self.loss, self.device)
        self.lr_finder.reset()  # Reset the model and optimizer to their initial state

    def scheduleOCP(self,maxlr):
        self.suggested_lr = maxlr    # this is taken from LRfinder suggested LR
        # Configure the One Cycle Policy parameters and scheduler
        self.ocp_parameters = {
            'max_lr': self.suggested_lr,
            'epochs': self.num_epochs,
            'steps_per_epoch': len(self.train_loader),
            'pct_start': self.pctabs / self.num_epochs,
            'div_factor': self.divfactor,
            'final_div_factor': self.finaldivfactor
        }
        self.scheduler = get_scheduler(self.optimizer, onecycle=self.onecyclestatus, **self.ocp_parameters)


    def runmymodel(self):
        for epoch in range(1, self.num_epochs + 1):
            # Iterate over each epoch
            _ = epochLR(epoch, self.scheduler, self.epochlr)  # Update learning rate for the epoch

            # Call the training function
            self.trainfunc(self.model, self.device, self.train_loader, self.optimizer, self.loss, self.scheduler, self.trainlosses, self.trainacc)

            # Evaluate the model on the test data loader
            self.missclassifiedimages = self.testfunc(self.model, self.device, self.test_loader, self.loss, self.testlosses, self.testacc)

    def modelplots(self):
        post_accuracyplots(self.trainlosses,self.testlosses,self.trainacc,self.testacc)

    def missclassplots(self,plottitle):
        showmisclassifiedsamples(self.missclassifiedimages,{v: k for k, v in self.labels.items()},plottitle,self.meanlist,self.stdlist)

    def showgramcam(self,plottitle):
        showgramcam(self.model,self.missclassifiedimages,{v: k for k, v in self.labels.items()},plottitle,self.meanlist,self.stdlist,self.device)










if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    print()
    # Intialize class instance for assignment10 methods and variables
    S11assignment = A11()
    print("Hyperparamaters & Methods Initialzed... \n")

    
    ####### Set Seed ##########
    torch.manual_seed(S11assignment.SEED)
    if S11assignment.device == 'cuda':
        torch.cuda.manual_seed(S11assignment.SEED)

    ###### Load Dataset #######
    S11assignment.loadmydata()
    print()


















############################ Rough Code to be ignored #############################
'''
    
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
'''






