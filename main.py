from DataLoader import *
from Models import *
from Train import *
from Utils import *

class A10():
    def __init__(self):
        # Initializing various hyperparameters for the model training
        self.batchsize = 512             # Batch size for the data loader
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
        self.loader = get_CIFARdataset_with_loader  # Function to get the CIFAR dataset with data loader
        self.trainfunc = train_ocp       # Training function
        self.testfunc = test             # Testing function
        self.device = setdevice()        # Set device to GPU if available, else CPU
        self.pctabs = 5                  # Percentage of total steps where the learning rate increases in One Cycle Policy
        self.divfactor = 10              # Initial division factor for the learning rate in One Cycle Policy
        self.finaldivfactor = 1          # Final division factor for learning rate in One Cycle Policy
        
        # Variables to store data for plotting accuracy and loss graphs
        self.trainlosses = []
        self.testlosses = []
        self.trainacc = []
        self.testacc = []
        self.epochlr = []
        
    def loadmydata(self):
        # Load dataset and its loader
        # Load test data
        _, self.test_loader = self.loader('test', self.loaderconfig())
        # Load train data
        self.labels, self.train_loader = self.loader('train', self.loaderconfig())

        print("Dataset loaded...")

    def loadmymodel(self):
        # Load the model and move it to the configured device
        self.model = self.modelclass(dropout_value=self.dropoutnum).to(self.device)
        print("Model Initialized...")

    def giveoptimizer(self):
        # Set up the optimizer with specified algorithm, learning rate, and weight decay
        self.optimizer = get_optimizer(self.model, self.optimizeralgo, self.lr, self.weightdecay)

    def giveloss(self):
        # Define the loss function based on the specified criteria
        self.loss = get_loss(self.losscriteria)

    def LRfinder(self):
        # Use a Learning Rate Finder to determine the optimal maximum learning rate
        lr_finder = initiateLRfinder(self.train_loader, self.model, self.optimizer, self.loss, self.device)
        self.suggested_lr = lr_finder.history["lr"][lr_finder.history["loss"].index(lr_finder.best_loss)]
        lr_finder.reset()  # Reset the model and optimizer to their initial state

    def scheduleOCP(self):
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

    def loaderconfig(self):
        # Create a dictionary of DataLoader configuration options
        kwargs = {
            'batch_size': self.batchsize,   # Set the batch size
            'shuffle': True,                # Shuffle data to ensure diverse batches
            'num_workers': self.numworkers, # Number of worker threads for data loading
            'pin_memory': True              # Enable pinning memory for faster GPU transfers
        }
        return kwargs

    def runmymodel(self):
        for epoch in range(1, self.num_epochs + 1):
            # Iterate over each epoch
            _ = epochLR(epoch, self.scheduler, self.epochlr)  # Update learning rate for the epoch

            # Call the training function
            self.trainfunc(self.model, self.device, self.train_loader, self.optimizer, self.loss, self.scheduler, self.trainlosses, self.trainacc)

            # Evaluate the model on the test data loader
            self.missclassifiedimages = self.testfunc(self.model, self.device, self.test_loader, self.loss, self.testlosses, self.testacc)

    


if __name__ == '__main__':

    # Intialize class instance for assignment10 methods and variables
    S10assignment = A10()
    print("Hyperparamaters & Methods Initialzed...")

    
    ####### Set Seed ##########
    torch.manual_seed(S10assignment.SEED)
    if S10assignment.device == 'cuda':
        torch.cuda.manual_seed(S10assignment.SEED)






















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






