import os

def make_save_dir(task_name, attn_type):
    save_dir = f'./results/{task_name}/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir