prompt_unknown_0s = """You are helping to generate the reasoning for a logical reasoning problem. The problem has no sufficient evidence to conclude if the question is proved or disproved, thus is not provable, or unknown. You will need to generate refutation or exploration reasoning steps to show that there is no evidence to draw a conclusion. Make sure your answer is clear and concise. Always end your answer by saying So the statement "..." is not provable and the answer is "unknown."
Question: {} This quesion is not provable. Please generate the reasoning steps. Make sure your answer is clear and concise.
Answer:
"""

prompt_unknown = """You are helping to generate the reasoning for a logical reasoning problem. The problem has no sufficient evidence to conclude if the question is proved or disproved, thus is not provable, or unknown. You will need to generate refutation or exploration reasoning steps to show that there is no evidence to draw a conclusion. Make sure your answer is clear and concise. Always end your answer by saying So the statement "..." is not provable and the answer is "unknown".
Example:
Quesiton: A few players are playing a boardgame. The current state of the game is as follows. The cheetah needs support from the swordfish. The crocodile winks at the moose. The dog burns the warehouse of the raven. The gecko needs support from the aardvark. The kudu learns the basics of resource management from the aardvark. The leopard eats the food of the penguin. The leopard holds the same number of points as the lion. The viperfish winks at the penguin. The zander respects the doctorfish. The hare does not attack the green fields whose owner is the hummingbird. The rabbit does not knock down the fortress of the sea bass. And the rules of the game are as follows. Rule1: Be careful when something eats the food that belongs to the penguin and also needs the support of the lion because in this case it will surely sing a victory song for the whale (this may or may not be problematic). Rule2: If the gecko needs the support of the aardvark and the kudu prepares armor for the aardvark, then the aardvark will not wink at the oscar. Rule3: If the dog burns the warehouse that is in possession of the raven, then the raven shows her cards (all of them) to the hare. Rule4: If you are positive that one of the animals does not knock down the fortress that belongs to the sun bear, you can be certain that it will wink at the oscar without a doubt. Rule5: If the leopard sings a victory song for the whale, then the whale attacks the green fields of the tilapia. Rule4 is preferred over Rule2. Based on the game state and the rules and preferences, does the whale attack the green fields whose owner is the tilapia? This quesion is not provable. Please generate the reasoning steps. Make sure your answer is clear and concise.
Answer: 
We know the whale's attack on the green fields of the tilapia depends on whether the leopard sings a victory song for the whale, as stated in Rule 5.
For the leopard to sing the victory song, it must both eat the food of the penguin and need the support of the lion, per Rule 1. 
However, the game state does not confirm whether the leopard needs the lion's support, leaving this condition unresolved. 
As none of the other rules (Rules 2, 3, or 4) are relevant to the whale's attack or the leopard's behavior, and there is insufficient information to verify the key condition in Rule 1, the conclusion remains unknown. 
So the statement "the whale attack the green fields whose owner is the tilapia" is not provable and the answer is "unknown".
Question: {} This quesion is not provable. Please generate the reasoning steps. Make sure your answer is clear and concise.
Answer:
"""