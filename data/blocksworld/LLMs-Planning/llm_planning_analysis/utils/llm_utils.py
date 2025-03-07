from transformers import StoppingCriteriaList, StoppingCriteria
import openai
from openai import OpenAI

client = OpenAI()
import os
# openai.api_key = os.environ["OPENAI_API_KEY"]
def generate_from_bloom(model, tokenizer, query, max_tokens):
    encoded_input = tokenizer(query, return_tensors='pt')
    stop = tokenizer("[PLAN END]", return_tensors='pt')
    stoplist = StoppingCriteriaList([stop])
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=max_tokens,
                                      temperature=0, top_p=1)
    return tokenizer.decode(output_sequences[0], skip_special_tokes=True)


def send_query(query, engine, max_tokens, model=None, stop="[STATEMENT]"):
    max_token_err_flag = False
    if engine == 'bloom':

        if model:
            response = generate_from_bloom(model['model'], model['tokenizer'], query, max_tokens)
            response = response.replace(query, '')
            resp_string = ""
            for line in response.split('\n'):
                if '[PLAN END]' in line:
                    break
                else:
                    resp_string += f'{line}\n'
            return resp_string
        else:
            assert model is not None
    elif engine == 'finetuned':
        if model:
            try:
                response = client.completions.create(model=model['model'],
                prompt=query,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["[PLAN END]"])
            except Exception as e:
                max_token_err_flag = True
                print("[-]: Failed GPT3 query execution: {}".format(e))
            text_response = response.choices[0].text if not max_token_err_flag else ""
            return text_response.strip()
        else:
            assert model is not None
    elif '_chat' in engine:
        
        eng = engine.split('_')[0]
        # print('chatmodels', eng)
        messages=[
        {"role": "system", "content": "You are the planner assistant who comes up with correct plans."},
        {"role": "user", "content": query}
        ]
        try:
            response = client.chat.completions.create(model=eng, messages=messages, temperature=0)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))
        text_response = response.choices[0].message.content if not max_token_err_flag else ""
        return text_response.strip()        
    else:
        try:
            response = client.completions.create(model=engine,
            prompt=query,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))

        text_response = response.choices[0].text if not max_token_err_flag else ""
        return text_response.strip()
    

def send_query_with_feedback(query, engine, messages=[]):
    err_flag = False
    context_window_hit = False
    rate_limit_hit = False
    if '_chat' in engine:
        eng = engine.split('_')[0]
        print('chatmodels', eng)
        if len(messages) == 0:
            messages=[
            {"role": "system", "content": "You are the planner assistant who comes up with correct plans."},
            {"role": "user", "content": query}
            ]
        else:
            #Just for validation message - query consists of the validation message
            messages.append({"role": "user", "content": query})
        try:
            response = client.chat.completions.create(model=eng, messages=messages, temperature=0)
        except openai.RateLimitError:
            err_flag = True
            rate_limit_hit = True
        except Exception as e: 
            err_flag = True
            if "maximum context length" in str(e):
                context_window_hit = True
            print("[-]: Failed GPT3 query execution: {}".format(e))
        text_response = "" if err_flag else response.choices[0].message.content
        messages.append({"role": "assistant", "content": text_response})
        return text_response.strip(), messages, context_window_hit, rate_limit_hit   
    else:
        raise Exception("[-]: Invalid engine name: {}".format(engine))
    

def save_gpt3_response(planexecutor, response, file):
    action_list = list(planexecutor.model["domain"].keys())
    action_list = [act.lower() for act in action_list]
    plan = []
    for line in response.split('\n'):
        if '[PLAN END]' in line:
            break
        else:
            action = line[line.find("(")+1:line.find(")")]
            if not action.strip():
                continue
            act_name = action.strip().split()[0]    
            if act_name and act_name.lower() in action_list:
            #find elements between ()
                plan.append(f'({action})')
    with open(file, 'w') as f:
        f.write('\n'.join(plan))
    return '\n'.join(plan)