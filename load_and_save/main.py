import torch
import torch.nn as nn

### Complete model ###
# torch.save(model,PATH) # use pickle model to save in seialized formate

# model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()


### State Dict ###
#torch.save(model.state_dict(), PATH)

# model must be created again with parameters
# model =  Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()


# model.load_state_dict(arg)