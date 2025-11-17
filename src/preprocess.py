import pandas as pd
import re

def gpt_summary_parser(summary):
    """
    The function splits the text in the 'gpt3.5_summary' atrtibute into a question list and an answer list
    
    Args:
        summary (str): the gpt3.5_summary attribute of the dataset

    Returns:
        question_list, answer_list (list): list of questions and answers from the summary (gpt3.5 summary)

    """

    if not isinstance(summary, str):
        return [],[]
    
    parts = re.split(r'The response provides[^:]*:\s*', summary, maxsplit=1, flags=re.IGNORECASE)
    
    if len(parts) < 2:
        return [],[]
    
    q_block, rest = parts
    a_block = re.split(f'\bOverall\b', rest, maxsplit=1, flags=re.IGNORECASE)[0]

    q_list = [m.strip() for m in re.findall(r'\d+\.\s*(.+)', q_block)]
    a_list = [m.strip() for m in re.findall(r'\d+\.\s*(.+)', a_block)]

    return q_list, a_list



def main(train_test):

    if train_test == 'training':
        df = pd.read_csv('./dataset/training_data.csv')
    elif train_test == 'test':
        df = pd.read_csv('./dataset/test_data.csv')
    else:
        return print('Enter valid argument')
    
    for i in [1, 2, 3]:
        annotator = 'annotator' + str(i)
        df = df.drop(annotator, axis=1)

    df = df.drop('url', axis = 1)
    df['answer'] = None

    for summary, group in df.groupby('gpt3.5_summary'):
        q_list, a_list = gpt_summary_parser(summary)

        if not q_list or not a_list:
            continue

        mapping = {
            q.strip(): a.strip() for q, a in zip(q_list, a_list) 
        }

        for idx, row in group.iterrows():
            q_text = str(row['question']).strip()
            subans = mapping.get(q_text)
            df.at[idx, 'answer'] = subans

    df.to_csv(f'./dataset/{train_test}_data_processed.csv')


    


if __name__ == '__main__':
    main('training')
    main('test')

