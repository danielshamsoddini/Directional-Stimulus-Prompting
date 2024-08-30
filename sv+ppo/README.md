Supervised+PPO Negotiation Agent

Training:
    python basic_train.py

PPO:
    python dialog_and_reinforce.py, the default arguments are what I have been using


Current Issues:
Over time in RLtraining, the model just ends up repeating text
    INFO:root:{'partner_agent': 'Hello!'}
    INFO:root:{'reinforce_agent': "Hello! I'm looking forward to this camping trip. I'm looking forward to it."}
    INFO:root:{'partner_agent': 'Are you sure you are going camping?'}
    INFO:root:{'reinforce_agent': "I'm going camping with my family. I'm going to be camping with my family."}
    INFO:root:{'partner_agent': 'Yes. That would be fun! I was hoping we can make a deal that works for you?'}
    INFO:root:{'reinforce_agent': "I'm going camping with my family. I'm going to be camping with my family. I'm going to be camping with my family."}
    INFO:root:{'partner_agent': 'Yes. That sounds like a great idea. I would appreciate extra food for your trip.'}
    INFO:root:{'reinforce_agent': "I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family."}
    INFO:root:{'partner_agent': 'I can help you!'}
    INFO:root:{'reinforce_agent': "I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family."}
    INFO:root:{'partner_agent': 'I would appreciate a bit more food. We can make a deal, though.'}
    INFO:root:{'reinforce_agent': "I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I'm going to be camping with my family. I"}
    INFO:root:{'partner_agent': 'I would appreciate 2 of the extra food. But I will be camping with my family.'}

The agent still seems too eager to agree to the partner

