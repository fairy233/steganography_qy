class StatusManager():
    def __init__(self):
        # data status
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # training status
        self.epoch = 0
        self.iter = 0
        self.optimizer = None
        self.scheduler = None
        self.print_str = ""

        # val status

        # time status

        # model status
        self.best_model_name = 'None'
        self.loss_meters = None
        self.loss_meter_names = None