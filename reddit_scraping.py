import praw  # https://praw.readthedocs.io/
from praw.models import MoreComments

from sentence_transformers import SentenceTransformer  # https://sbert.net/

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, tqdm_notebook  # https://tqdm.github.io/
from datetime import datetime  # https://docs.python.org/3/library/datetime.html
import os.path  # https://docs.python.org/3/library/os.path.html


# Input: submission class from PRAW
# Output: DataFrame with comment ID, author, time, body, and score
def get_comments_from_submission(submission):
    submission.comments.replace_more(limit=None)
    submission_list = submission.comments.list()
    comment_body = []
    comment_time = []
    comment_id = []
    comment_author = []
    comment_score = []
    submission_id = []

    for comment in submission_list:
        if comment.body != '[deleted]':
            comment_body.append(comment.body.replace('\n',''))
            comment_time.append(comment.created_utc)
            comment_id.append(comment.id)
            comment_author.append(comment.author)
            comment_score.append(comment.score)
            submission_id.append(comment.submission.id)

    data = {'Submission_ID': submission_id,
            'ID': comment_id,
            'Author': comment_author,
            'Time': comment_time,
            'Comment': comment_body,
            'Score': comment_score}

    return pd.DataFrame(data)


# Input: subreddit = subreddit name,
#        n = number of submissions
# Output: list of submissions
def get_hot_submissions(subreddit, n):
    data = []
    for submission in reddit.subreddit(subreddit).hot(limit=n):
        data.append(submission)

    return data


def get_top_submissions(subreddit, n, time):
    data = []
    for submission in reddit.subreddit(subreddit).top(limit=n, time_filter = time):
        data.append(submission)

    return data


def get_submissions(subreddit_name: str, type: str, n: int, time_filter: str = '', time1=0, time2=float('inf')):
    type_valid = ['hot', 'top', 'new']
    time_filter_valid = {"all", "day", "hour", "month", "week", "year", ''}

    if type not in type_valid:
        raise ValueError("save_df: type must be one of %r." % type_valid)

    if time_filter not in time_filter_valid:
        raise ValueError("save_df: time_filter must be one of %r." % time_filter_valid)

    func_valid = [reddit.subreddit(subreddit_name).hot, reddit.subreddit(subreddit_name).top, reddit.subreddit(subreddit_name).new]
    func = func_valid[type_valid.index(type)]

    data = []
    if type == 'hot' or type == 'new':
        for submission in func(limit=n):
            if time1 <= submission.created_utc <= time2:
                data.append(submission)
    if type == 'top':
        for submission in func(limit=n, time_filter=time_filter):
            if time1 <= submission.created_utc <= time2:
                data.append(submission)

    return data


# Input: list of submissions
# Output: all comments of all submissions
def get_all_comments(submissions_list):
    data = [get_comments_from_submission(s) for s in submissions_list]  # list of DataFrames
    return pd.concat(data, ignore_index=True)


def load_save_df(subreddit: str, type: str, n: int, time_filter: str = '', time1 = None, time2 = None):
    if type == 'hot':
        name = 'Data/' + subreddit + '_' + type + '_' + str(n) + '.csv'
        if os.path.exists(name):
            df = pd.read_csv(name)
        else:
            submissions = get_hot_submissions(subreddit, n)
            df = get_all_comments(submissions)
            df.to_csv(name, index = False)

    if type == 'top':
        name = 'Data/' + subreddit + '_' + type + '_' + time_filter  + '_' + str(n) + '.csv'
        if os.path.exists(name):
            df = pd.read_csv(name)
        else:
            submissions = get_hot_submissions(subreddit, n)
            df = get_all_comments(submissions)
            df.to_csv(name, index = False)

    return df


def create_master_df(subreddit_name: str, type: str, n: int, time_filter: str = '', time1=0, time2=float('inf')):
    path = 'Data/' + subreddit_name + '.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        submission_id = set(df['Submission_ID'].tolist())
    else:
        submission_id = set([])

    data = []
    new_submissions = get_submissions(subreddit_name, type, n, time_filter, time1, time2)
    for submission in new_submissions:
        print(new_submissions.index(submission))
        if submission.id not in submission_id:
            data.append(get_comments_from_submission(submission))

    if not data:
        print('Data is already in the csv file.')
    else:
        new_df = pd.concat(data, ignore_index=True)

        if os.path.exists(path):
            new_df.reset_index()
            new_df.to_csv(path, mode='a', index=False, header=False, encoding='utf-8-sig')
        else:
            new_df.to_csv(path, index = False, encoding='utf-8-sig')

        print('Data uploaded to csv file.')


reddit = praw.Reddit("bot1")

create_master_df('apple', 'hot', 2000)
