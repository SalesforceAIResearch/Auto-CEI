# Data generation for Blocksworld

1. Clone the github repo https://github.com/aibasel/downward to the folder `/data/blocksworld/LLMs-Planning/planner_toosls`

2. Run the command using script `script/blocksworld_gen_data.sh`:
```bash
pip install PyYAML tarski pddl
cd ./data/blocksworld/LLMs-Planning/planner_tools/downward
./build.py
cd ../../llm_planning_analysis
export FAST_DOWNWARD=../planner_tools/downward
export PR2=../planner_tools/PR2/pr2plan
export VAL=../planner_tools/VAL
python3 problem_generators.py --config blocksworld_large
python3  training_data_generation.py --config blocksworld_large
cd ../../../../
python3 data/blocksworld/LLMs-Planning/llm_planning_analysis/data_pre.py  
```