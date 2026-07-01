Mohammadreza RasouliSadr [40217693]
Parham Moafi [40221603]
Radmehr Soleimanian [40218473]





trained agent : models/dqn_trained.pt
training analysis and plots : docs/report.ipynb
evaluation metrics : Evaluation/evaluation_report.json


TO VISUALIZE THE TRAINED AGENT (Task 8 Requirement)
Run the following command to render the MetaDrive environment and
watch the trained agent navigate successfully:
python visualize.py --mode --episodes --seed --scenarios
""" 
parser.add_argument("--mode", type=str, default="3D", choices=["3D", "top_down"])
parser.add_argument("--episodes", type=int, default=100, help="number of attempts to watch")
parser.add_argument("--seed", type=int, default=51, help="starting map seed for testing")
parser.add_argument("--scenarios", type=int, default=100, help="number of unique maps to cycle through")
"""


TO RUN THE EVALUATION
To test the zero-shot generalization of agent on completelyunseen maps
python evaluate.py --episodes --seed --scenarios
"""
parser.add_argument("--episodes", type=int, default=100, help="number of evaluation episodes")
parser.add_argument("--seed", type=int, default=51, help="starting map seed")
parser.add_argument("--scenarios", type=int, default=100, help="number of different maps to test")
"""
(this will output the success/crash/out-of-road rates)


TO TRAIN THE AGENT
If you wish to train a new model, you can run main.py and pass the
training parameters via command line arguments.
python main.py --episodes --seed --scenarios
"""
parser.add_argument("--episodes", type=int, default=25000, help="Number of training episodes")
parser.add_argument("--seed", type=int, default=1, help="starting map seed")
parser.add_argument("--scenarios", type=int, default=50, help="number of different maps to train on")
"""