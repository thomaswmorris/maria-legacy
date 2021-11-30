

import maria 

def default_simulation():
    "Simulate the default model."
    default_model = maria.model()

    default_model.sim()

    assert 1==2
    
    
    
def ACT_simulation():
    "Simulate a ten-minute ACT scan."
    default_model = maria.model()

    default_model.sim()   