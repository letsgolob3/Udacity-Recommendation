# -*- coding: utf-8 -*-
"""

"""


#Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import project_tests as t
import pickle



df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()

# Show df_content to get an idea of the data
df_content.head()


'''
What are the columns and datatypes?

Are there missing values?
'''
 
df.info()

df_content.info()

'''
df
    - article id | float | No missing values
    - title | object | No missing values
    - email | object | There are missing values
    
df_content
    - doc_body | object | There are missing values
    - doc_description | object | There are missing values
    - doc_full_name | object | No missing values
    - doc_status | article_id | No missing values
    
df and df_content can be linked via article_id

Now lets answer some additional questions and do some additional 
exploration:
    
Emails:
    How many unique email hashes are there?
    Are the email hashes all of fixed length?
'''

# Unique email hashes 
n_emails=df['email'].nunique()
print(f'There are {n_emails} unique emails')

# Email hashes all of fixed length?
df['len_email']=df['email'].str.len()

print(df['len_email'].value_counts(dropna=False))

df.drop(columns=['len_email'],inplace=True)

'''
There are 17 missing emails and each email is a 40 character hash.
If we are going to group by similar 40 character hashes, that may
take some significant processing power.  Lets convert those to integers.
The number of unique email_ids should be the same as unique emails.  
'''

df['email_id']=df.groupby('email').ngroup()


# Unique email hashes 
n_email_ids=df['email'].nunique()
print(f'There are {n_email_ids} unique email ids')

try:
    assert n_emails==n_email_ids
except AssertionError as e:
    print(e)

#What is the email id from the emails that are null?
email_id_null=df.loc[df.email.isnull(),'email_id'].unique()[0]
print(f'Email id for emails that are null is {email_id_null}')


'''
1. What is the distribution of how many articles a user interacts with in the dataset? 
Provide a visual and descriptive statistics to assist with giving 
a look at the number of times each user interacts with an article.

'''

# Do users interact with the same article more than once?
same_article_by_user=df.groupby(['email_id','article_id'])\
                       .agg({'title':'count'})\
                       .reset_index().rename(columns={'title':'num_views'})\
                       .query('email_id>=0 & num_views>1')

same_article_by_user.head()  

'''
Yes, some users interact with the same article more than once.
For example, we see from above that the user with the email_id of 0
viewed article id 43 twice.

When considering the distribution of how many articles a user interacts with,
we will consider any interaction here even if a user interacted with the same 
article more than once.  However, the distribution would look slightly different
if only considering unique articles by user.  

Remove the email_ids that are null
'''

unique_articles_by_user=df.groupby('email_id').agg({'article_id':'nunique'})\
                                              .reset_index()\
                                              .query('email_id>=0')  

total_art_by_user=df.groupby('email_id').agg({'article_id':'count'})\
                                              .reset_index()\
                                              .query('email_id>=0')\
                                              .rename(columns={'article_id':'interaction_count'})

print(total_art_by_user.describe())

print(total_art_by_user['interaction_count'].value_counts())

total_art_by_user['interaction_count'].hist(bins=30)
plt.title('Distribution of number of user-article interactions')
plt.ylabel('count')
plt.show()           

total_art_by_user['bin']=pd.cut(total_art_by_user['interaction_count'],
                                bins=[0,10,20,30,40,50,100,200,300,
                                      total_art_by_user['interaction_count'].max()])

bin_stats=total_art_by_user.groupby('bin').agg({'interaction_count':'count'}).reset_index()
bin_stats['pct_interactions']=round(bin_stats['interaction_count']/bin_stats['interaction_count'].sum(),4)*100
print(bin_stats)

'''
For the 5148 users with an email:
    - The average number of interactions is 9
    - The maximum number of interactions is 364
    - The minimum number of interactions is 1
    - The median number of interactions is 3
    - The mode number of interactions is 1

The distribution is right skewed as the majority of users interacted with
fewer than 50 articles.  Over 78% of users interacted with 10 articles or less and 
over 97% of users interacted with 50 articles or less.  As is the case with right-skewed
data, the mode (1) is less than the median (3), which is less than the mean (9).  

There were two users that had over 300 interactions.  
'''




# Fill in the median and maximum number of user_article interactios below
total_art_by_user.describe()

int(total_art_by_user['interaction_count'].median())


median_val = total_art_by_user['interaction_count'].median() 
print(f'# 50% of individuals interact with {median_val} number of articles or fewer.')

max_views_by_user = median_val = total_art_by_user['interaction_count'].max()
print(f'# The maximum number of user-article interactions by any 1 user is {max_views_by_user}.')
                                              





#2. Explore and remove duplicate articles from the df_content dataframe.

# Find and explore duplicate articles

duplicates=df_content.loc[df_content.duplicated()]

print(f'# There are {len(duplicates)} duplicate records within the df_content dataframe.')  


# Remove any rows that have the same article_id - only keep the first
df_content['art_rank']=df_content.groupby('article_id')['article_id'].rank(method='first')

num_dup_art=len(df_content.loc[df_content['art_rank']>1])

print(f'There are {num_dup_art} articles with duplicate ids in the data.')

df_content=df_content.loc[df_content['art_rank']==1].copy().drop(columns=['art_rank'])



'''
3. Use the cells below to find:

a. The number of unique articles that have an interaction with a user.
b. The number of unique articles in the dataset (whether they have any interactions or not).
c. The number of unique users in the dataset. (excluding null values)
d. The number of user-article interactions in the dataset.

'''

df['article_id']=pd.to_numeric(df['article_id'],downcast='integer')
df_content['article_id']=pd.to_numeric(df_content['article_id'],downcast='integer')


# a 
unique_articles = df['article_id'].nunique()

# b
articles_in_df=set(df['article_id'].unique())
articles_in_dfcontent=set(df_content['article_id'].unique())

# There are article ids in df not in df_content
# There are article ids in df_content not in df
in_df_not_content=articles_in_df.difference(articles_in_dfcontent)
in_content_not_df=articles_in_dfcontent.difference(articles_in_df)

total_articles = len(articles_in_df.union(articles_in_dfcontent))


# c
unique_users = df['email_id'].nunique()

# d 



'''
4. Use the cells below to find the most viewed article_id, 
as well as how often it was viewed. 
After talking to the company leaders, 
the email_mapper function was deemed a reasonable way to map users to ids. 
There were a small number of null values, and it was found that all of these 
null values likely belonged to a single user 
(which is how they are stored using the function below).
'''

# The most viewed article in the dataset as a string with one value following the decimal 

most_viewed_article_id = str(round(float(df['article_id'].mode()[0]),1))

# The most viewed article in the dataset was viewed how many times?
max_views = len(df.loc[df['article_id']==df['article_id'].mode()[0]])



def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded


#Rank-based recommendations 


test2=df.groupby('article_id').agg({'article_id':'count','title':'first'})\
                              .rename(columns={'article_id':'counts'}).reset_index()\
                              .sort_values('counts',ascending=False)    


#User-User based collaborative filtering

# Since the above function was provided, remove the email_id column
df.drop(columns='email_id',inplace=True)


df=df.assign(interaction=1).copy()

user_item=pd.pivot_table(df,index='user_id',columns='article_id',values='interaction')


user_item.fillna(0,inplace=True)



def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    article_counts=df.groupby('article_id').agg({'article_id':'count','title':'first'})\
                                           .rename(columns={'article_id':'counts'}).reset_index()\
                                           .sort_values('counts',ascending=False)    


    top_articles=list(article_counts['title'].unique())[0:n]
    
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    article_counts=df.groupby('article_id').agg({'article_id':'count','title':'first'})\
                                           .rename(columns={'article_id':'counts'}).reset_index()\
                                           .sort_values('counts',ascending=False)    


    top_articles=list(article_counts['article_id'].unique())[0:n]
    
 
    return top_articles # Return the top article ids

print(get_top_articles(10))
print(get_top_article_ids(10))


#find_similar_users function

test=user_item.head(5)

dot_prods={}

user1=np.asarray(test.loc[test.index==1])


for user_id in test.index:
    
    userN=np.asarray(test.loc[test.index==user_id].transpose())
    
    dot_prods[user_id]=np.dot(user1,userN)[0,0]
    

dot_prod_df=pd.DataFrame.from_dict(dot_prods,orient='index').reset_index()\
                        .rename(columns={'index':'user_id',0:'dot_prod'})\
                        .sort_values('dot_prod',ascending=False)


most_similar_users=list(dot_prod_df['user_id'].unique())

most_similar_users.pop(1)



def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    
    # compute similarity of each user to the provided user
    dot_prods={}
    
    user_input=np.asarray(user_item.loc[user_item.index==user_id])
    
    for userID in user_item.index:

        userN=np.asarray(user_item.loc[user_item.index==userID].transpose())

        dot_prods[userID]=np.dot(user_input,userN)[0,0]    
    

    # sort by similarity
    dot_prod_df=pd.DataFrame.from_dict(dot_prods,orient='index').reset_index()\
                        .rename(columns={'index':'user_id',0:'dot_prod'})\
                        .sort_values('dot_prod',ascending=False)
    

    # create list of just the ids
    most_similar_users=list(dot_prod_df['user_id'].unique())
    
   
    # # remove the own user's id
    most_similar_users.remove(user_id)
       
    return most_similar_users # return a list of the users in order from most to least similar


print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))


# test=user_item.loc[user_item.index.isin([46,23])]



# article_ids=['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0']

# article_ids=list(pd.to_numeric(article_ids,downcast='integer'))

# test=list(df.loc[df['article_id'].isin(article_ids),'title'].copy().unique())

def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    
    article_ids=list(pd.to_numeric(article_ids,downcast='integer'))
    
    article_names=list(df.loc[df['article_id'].isin(article_ids),'title'].copy().unique())
    
    
    return article_names # Return the article names associated with list of article ids


assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."





user_id=1

#This subsets the user_item on user ID, then transposes it to one column
#and filters all the 1s.  Then it gets all the article ids for each 1 and 
#puts it into a unique list of article ids.  
test_art_ids=list(set(user_item.loc[user_item.index==user_id].copy().transpose()\
                                                     .rename(columns={1:'read_indicator'})\
                                                     .query('read_indicator>0')\
                                                     .reset_index()['article_id']))

test_art_names=get_article_names(test_art_ids)
    
    

def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    article_ids=list(set(user_item.loc[user_item.index==user_id].copy().transpose()\
                                                         .rename(columns={user_id:'read_indicator'})\
                                                         .query('read_indicator>0')\
                                                         .reset_index()['article_id']))
        
    article_ids=[str(float(num)) for num in article_ids]
    
    article_names=get_article_names(article_ids)
    
    return article_ids, article_names # return the ids and names

test1,_=get_user_articles(20)



def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    
    recs=[]
    
    closest_users=find_similar_users(user_id)
    
    article_ids_seen_user,_=get_user_articles(user_id)
    
    for close_nbr in closest_users:

        article_ids_seen_closest,_=get_user_articles(close_nbr)

        # Articles seen by neighbor but not the user
        new_articles=list(set(article_ids_seen_closest).difference(set(article_ids_seen_user)))
        
        recs.extend(new_articles)
        
        if len(recs)>=m:
            recs=recs[:m]
            break
            

    return recs


test2=user_user_recs(1, 10)
    
    
    
def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    
    # Obtain number of articles viewed by all users
    num_ints=df.groupby('user_id').agg({'article_id':'count'}).reset_index()\
                                  .rename(columns={'article_id':'num_interactions'})
    
    
    # compute similarity of each user to the provided user
    dot_prods={}
    
    user_input=np.asarray(user_item.loc[user_item.index==user_id])
    
    for userID in user_item.index:

        userN=np.asarray(user_item.loc[user_item.index==userID].transpose())

        dot_prods[userID]=np.dot(user_input,userN)[0,0] 
        

    dot_prod_df=pd.DataFrame.from_dict(dot_prods,orient='index').reset_index()\
                        .rename(columns={'index':'neighbor_id',0:'similarity'})\
                        .sort_values('similarity',ascending=False)
    
    # Create the neighbors_df with neighbor_id, similarity, & num_interactions
    neighbors_df=pd.merge(dot_prod_df,num_ints,left_on='neighbor_id',
                          right_on='user_id',how='inner',
                          validate='one_to_one')
    
    # Sort by similarity
    neighbors_df.sort_values(by=['similarity','num_interactions'],
                            ascending=[False,False],inplace=True)
    
    # Remove the user_id column; it is redundant with the neighbor_id column
    neighbors_df.drop(columns=['user_id'],inplace=True)
    
    # Remove similarity to of user with themselves
    neighbors_df=neighbors_df.loc[neighbors_df['neighbor_id']!=user_id].copy()
    

    return neighbors_df # Return the dataframe specified in the doc_string

test3=get_top_sorted_users(1)




user_id=1

def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    recs=[]
    
    # This is the arbitrary part of choosing a similar user
    # closest_users=find_similar_users(user_id)
    
    # Choose closest users based on total article interactions 
    closest_users=list(get_top_sorted_users(user_id)['neighbor_id'])
    
    article_ids_seen_user,_=get_user_articles(user_id)
    
    for close_nbr in closest_users:

        article_ids_seen_closest,_=get_user_articles(close_nbr)
        
        article_ids_seen_closest = [pd.to_numeric(idx,downcast='integer') for idx in article_ids_seen_closest]
        
        article_ids_seen_closest_df=pd.DataFrame(article_ids_seen_closest,columns=['nbr_art_id'])
        
        
        # Now instead of arbitrary article ids list from above, sort the article_ids
        # based on popular articles.
        article_counts=df.groupby('article_id').agg({'article_id':'count','title':'first'})\
                                               .rename(columns={'article_id':'counts'}).reset_index()\
                                               .sort_values('counts',ascending=False)[['article_id','counts']] 
        
        article_counts=pd.merge(article_counts,article_ids_seen_closest_df,left_on='article_id',
                                right_on='nbr_art_id',validate='one_to_one',how='left')
        
        article_ids_seen_closest_sorted=list(article_counts['nbr_art_id'].dropna())
        
        article_ids_seen_closest_sorted = [int(article) for article in article_ids_seen_closest_sorted]

        # Articles seen by neighbor but not the user
        new_articles=list(set(article_ids_seen_closest_sorted).difference(set(article_ids_seen_user)))
        
        recs.extend(new_articles)
        
        if len(recs)>=m:
            recs=recs[:m]
            break
 
    rec_names=get_article_names(recs)
    
    return recs, rec_names


test4_ids, test4_names= user_user_recs_part2(20, 10)
