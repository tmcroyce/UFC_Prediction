# cd OneDrive/Data Science/Personal_Projects/Sports/UFC_Prediction/notebooks/final_notebooks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sqlite3
import seaborn as sns
from matplotlib.pyplot import figure
from bs4 import BeautifulSoup
import time
import requests     # to get images
import shutil       # to save files locally
import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from random import randint
import  random
import os
#os.chdir('C:/Users/tmcro/OneDrive/Data Science/Personal_Projects/Sports/UFC_Prediction')
from cmath import nan
from bs4 import BeautifulSoup
import streamlit as st
import pickle

home = 'C:/Users/tmcro/OneDrive/Data Science/Personal_Projects/Sports/UFC_Prediction/data/'
home2 = 'C:/Users/tmcro/OneDrive/Data Science/Personal_Projects/Sports/UFC_Prediction/'
#------------------------------  Define Functions -----------------------------------------------------------------
# function to return the next 2 UFC events
def get_next_events(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # get events
    event1 = soup.find('div', class_='c-card-event--result__info')
    event1_txt = soup.find('div', class_='c-card-event--result__info').text
    event1_url = event1.find('a')['href']
    event1_url = 'https://www.ufc.com' + event1_url
    event1_title = event1_txt.split('\n')[1]
    event1_time = event1_txt.split('/')[1]

    data = pd.DataFrame({'event_title': [event1_title], 'event_url': [event1_url], 'event_date': [event1_time]})

    event2 = soup.find('div', class_='c-card-event--result__info').find_next('div', class_='c-card-event--result__info')
    event2_txt = soup.find('div', class_='c-card-event--result__info').find_next('div', class_='c-card-event--result__info').text
    event2_url = event2.find('a')['href']
    event2_url = 'https://www.ufc.com' + event2_url
    event2_title = event2_txt.split('\n')[1]
    event2_time = event2_txt.split('/')[1]


    data = data.append({'event_title': event2_title, 'event_url': event2_url, 'event_date': event2_time}, ignore_index=True)
    
    event3 = soup.find('div', class_='c-card-event--result__info').find_next('div', class_='c-card-event--result__info').find_next('div', class_='c-card-event--result__info')
    event3_txt = soup.find('div', class_='c-card-event--result__info').find_next('div', class_='c-card-event--result__info').find_next('div', class_='c-card-event--result__info').text
    event3_url = event3.find('a')['href']
    event3_url = 'https://www.ufc.com' + event3_url
    event3_title = event3_txt.split('\n')[1]
    event3_time = event3_txt.split('/')[1]

    data = data.append({'event_title': event3_title, 'event_url': event3_url, 'event_date': event3_time}, ignore_index=True)
    
    return data

# Function to get the fight card for a given event
def get_event_fights(event_url):
    page = requests.get(event_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # get main card, fight 1

    mcn = soup.find_all('li', class_='l-listing__item')
    # get num of mc
    num_mc = len(mcn)
    # for each mc, do the following
    data = pd.DataFrame()
    n = 0
    for i in mcn:
        mc = mcn[n]
        # fight 1
        fighter1= mc.find('div', class_ ='c-listing-fight__corner-name c-listing-fight__corner-name--red').text
        fighter1 = fighter1.replace('\n', ' ')
        fighter1 = fighter1.strip()
        fighter2 = mc.find('div', class_ ='c-listing-fight__corner-name c-listing-fight__corner-name--blue').text
        fighter2 = fighter2.replace('\n', ' ')
        fighter2 = fighter2.strip()
        weightclass = mc.find('div', class_='c-listing-fight__class-text').text
        fighter1_odds = mc.find('span', class_='c-listing-fight__odds').text
        fighter2_odds = mc.find('span', class_='c-listing-fight__odds').find_next('span', class_='c-listing-fight__odds').text
        fighter1_odds = fighter1_odds.replace('\n', '')
        fighter2_odds = fighter2_odds.replace('\n', '')
        # fighter odds to float
        if (fighter1_odds == '-') :
            fighter1_odds = nan
        if (fighter2_odds == '-') :
            fighter2_odds = nan

        data = data.append({'fighter1': fighter1, 'fighter2': fighter2, 'weightclass': weightclass, 
                            'fighter1_odds': fighter1_odds, 'fighter2_odds': fighter2_odds}, ignore_index=True)
        n = n + 1
    return data

# get next events if event fighter data is not na
def get_next_events2(url):
    data = get_next_events(url)
    for i in range(0, len(data)):
        event_url = data['event_url'][i]
        event_fights = get_event_fights(event_url)
        if (len(event_fights) == 0):
            data = data.drop(i)
    return data

def get_next_event_ufcstats():
    url = 'http://www.ufcstats.com/statistics/events/upcoming'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # get events
    event1 = soup.find('td', class_='b-statistics__table-col')
    event1_txt = soup.find('td', class_='b-statistics__table-col').text
    event_txt = event1_txt.replace('   ', '').replace('\n', '').strip()
    event_title = event_txt.split('  ')[0]
    event_date = event_txt.split('  ')[1]
    event1_url = event1.find('a')['href']
    data = pd.DataFrame({'event_title': [event_title], 'event_url': [event1_url], 'event_date': [event_date]})


    return data

def get_fighter_urls(event_details_url):
    page = requests.get(event_details_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # get events
    events = soup.find_all('tr', class_='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click')
    n = 0
    next_event_data = pd.DataFrame()

    for event in events:
        fighters = events[n].find_all('p', class_='b-fight-details__table-text')
        fighter1 = fighters[0].text
        fighter1 = fighter1.replace('  ', '').replace('\n', '').strip()
        fighter2 = fighters[1].text
        fighter2 = fighter2.replace('  ', '').replace('\n', '').strip()
        fighter1_url = fighters[0].find('a')['href']
        fighter2_url = fighters[1].find('a')['href']
        next_event_data = next_event_data.append({'fighter1' :fighter1, 'fighter2:' : fighter2, 'fighter1_url': fighter1_url, 'fighter2_url':fighter2_url, 'fight#' : n+1}, ignore_index = True)
        n += 1

    return next_event_data

def secret_number(event_url):
    if event_url == 'https://www.ufc.com/event/ufc-fight-night-november-05-2022':
        return '1119'
    if event_url == 'https://www.ufc.com/event/ufc-281':
        return '1114'
    if event_url == 'https://www.ufc.com/event/ufc-fight-night-november-19-2022':
        return '1122'
    if event_url == 'https://www.ufc.com/event/ufc-fight-night-december-03-2022':
        return '1123'

next_eventz = get_next_event_ufcstats()
##########           Data        ################

data = pd.read_csv(home + 'ufc_stats/agg2/all_fights_9_27_V9.csv')

##########           GET EVENTS       ################

# make sure events have fight info. If not, disregard that event
next = get_next_events2('https://www.ufc.com/events')

########           Select Next Event    ################

event = st.sidebar.selectbox('Select Event', next['event_title'])
selected_event = event
event_url =  next['event_url'][next['event_title'] == selected_event].values[0]
selected_event_secret_number = secret_number(event_url)

next_event = get_event_fights(event_url)
fight = st.sidebar.selectbox('Select Fight', next_event['fighter1'] + ' vs. ' + next_event['fighter2'])

## Get Names ##
selected_fighter_1 = fight.split(' vs. ')[0]
selected_fighter_2 = fight.split(' vs. ')[1].strip()

########          Scrape UFC.com Data    ################

# get the matchup fight numbers

page = requests.get(event_url)
soup = BeautifulSoup(page.content, 'html.parser')
h = soup.find_all('div', class_='c-listing-fight')
data_fmid = []
for i in h:
    data_fmid.append(i['data-fmid'])

next_event['fight_number'] = data_fmid[:len(next_event)]
next_event['matchup_url'] = event_url +'#' + next_event['fight_number'].astype(str)
selected_matchup_url = next_event['matchup_url'][next_event['fighter1'] == selected_fighter_1].values[0]

# Function to scrape UFC fight data
def grab_matchup_data(matchup_url):
    response = requests.get(matchup_url)
    soup = BeautifulSoup(response.text, 'html.parser').text
    soup = soup.replace('   ', '').replace('\n', '')
    
    od = soup.find('Odds')
    rec = soup.find('Record')
    a_record = soup[od + 5 : rec - 2]
    last_fight = soup.find("Last Fight")
    b_record = soup[rec + 7 : last_fight - 5]

    hite = soup.find('Height')
    f = soup.find("' ")
    a_height = soup[f -1 : hite - 2]
    # find second occurance of f
    f2 = soup.find("' ", f + 1)
    b_height = soup[hite + 7 : f2+5]

    # Find reach
    reach = soup.find('Reach')
    # find second occurance of "LB"
    lb = soup.find('LB')
    lb2 = soup.find('LB', lb + 1)
    a_reach = soup[lb2 +5 : reach ]
    inn = soup.find("in ")
    # get the word after reach
    big_space = soup.find('  ', reach + 1)
    b_reach = soup[reach + 6 : big_space]

    # Find Leg Reach
    leg = soup.find('Leg Reach')
    big_space2 = soup.find('  ', big_space + 1)
    a_leg = soup[big_space2 + 2 : leg]
    big_space4 = soup.find('  ', big_space2 + 2)
    b_leg = soup[leg + 10 : leg + 17]

    a_record = a_record.strip()
    b_record = b_record.strip()

    a_height_ft = float(a_height[:1])
    a_height_in = float(a_height[3:].replace("'", "").replace('"', ''))
    a_height = (a_height_ft * 12) + a_height_in 

    b_height_ft = float(b_height[:1])
    b_height_in = float(b_height[3:].replace("'", "").replace('"', ''))
    b_height = (b_height_ft * 12) + b_height_in

    a_reach = float(a_reach.replace(' in', '').strip())
    b_reach = float(b_reach.replace(' in', '').strip())
    a_leg = float(a_leg.replace(' in', '').strip())
    b_leg = float(b_leg.replace(' in', '').strip())
    
    return a_record, b_record, a_height, b_height, a_reach, b_reach, a_leg, b_leg

url = 'https://www.ufc.com/matchup/' + selected_event_secret_number + '/' + next_event[next_event['fighter1'] == selected_fighter_1]['fight_number'].values[0]

a_record, b_record, a_height, b_height, a_reach, b_reach, a_leg, b_leg = grab_matchup_data(url)

##########     Get Fighter Info      ############# 


# GET PICTURE URLS
fighter1_pic_url = "ufc.com/athlete/" + selected_fighter_1.replace(' ', '-').lower()
fighter2_pic_url = "ufc.com/athlete/" + selected_fighter_2.replace(' ', '-').lower()

# SCRAPE PICTURES FROM UFC.COM
def get_info(url):
    page = requests.get(url)
    return page.text

pagedata = get_info('https://' + fighter1_pic_url)
soup = BeautifulSoup(pagedata, 'html.parser')
fighter1_pic_url = soup.find('img', class_='hero-profile__image')['src']

page2data = get_info('https://' + fighter2_pic_url)
soup2 = BeautifulSoup(page2data, 'html.parser')
fighter2_pic_url = soup2.find('img', class_='hero-profile__image')['src']


################ FIGHTER INFO ####################
st.header('UFC Fight Prediction')
st.write('Choose a fight from the sidebar to see the prediction.')

f1 = data[data['Fighter_A'] == selected_fighter_1]
f2 = data[data['Fighter_A'] == selected_fighter_2]

# transpose f1
cols = ['Fighter_A', 'A_Height', 'A_Reach', 'A_Leg_Reach']
f11 = f1[cols].reset_index().drop(['index'], axis=1)
# rename columns
f11.columns = ['Fighter', 'Height', 'Reach', 'Leg_Reach']
f22 = f2[cols].reset_index().drop(['index'], axis=1)
f22.columns = ['Fighter', 'Height', 'Reach', 'Leg_Reach']


# function to convert vegas odds to implied probability
def odds_to_prob(odds):
    odds = float(odds)
    if (odds > 0):
        prob = round(1/(odds/100 + 1),3) * 100
        prob = str(round(prob, 3)) + '%'
        return prob
    else:
        prob = round(1 - 1/(-odds/100 + 1),3)*100
        prob = str(round(prob, 3)) + '%'
        return prob

# Assign Height and Length Values
try:
    dif = a_height - b_height
    dif2 = a_reach - b_reach
    dif3 = a_leg - b_leg


    col1, col2 = st.columns(2)
    with col1:
        st.subheader(selected_fighter_1)
        st.image(fighter1_pic_url, width=200)
        st.metric(label = 'Vegas Odds', value=next_event['fighter1_odds'][next_event['fighter1'] == selected_fighter_1].values[0])
        st.metric(label = 'Odds-Implied Probability', 
                    value=odds_to_prob(next_event['fighter1_odds'][next_event['fighter1'] == selected_fighter_1].values[0]))
        st.metric(label = 'Height', value=a_height, delta = dif)
        st.metric(label = 'Reach', value=a_reach, delta = dif2)
        st.metric(label = 'Leg Reach', value=a_leg, delta = dif3)

    with col2:
        st.subheader(selected_fighter_2)
        st.image(fighter2_pic_url, width=200)
        st.metric(label = 'Vegas Odds', value=next_event['fighter2_odds'][next_event['fighter2'] == selected_fighter_2].values[0])
        st.metric(label = 'Odds-Implied Probability', 
                    value=odds_to_prob(next_event['fighter2_odds'][next_event['fighter2'] == selected_fighter_2].values[0]))
        st.metric(label = 'Height', value=b_height, delta = -dif)
        st.metric(label = 'Reach', value=b_reach, delta = -dif2)
        st.metric(label = 'Leg Reach', value=b_leg, delta = -dif3)
except: 
    st.markdown('THERES A PROBLEM WITH ONE OF THE FIGHTERS GIVEN METRICS... CALL CUSTOMER SUPPORT OR SOMETHING')



###########  CALCULATE FIGHTER STATS FOR CURRENT FIGHT #############
# calculate differences between in-match stats

def calc_diffs(df):
    for col in df.columns:
        if 'A_' in col:
            new_col = col.replace('A_', 'Dif_')
            df[new_col] = df[col] - df[col.replace('A_', 'B_')]
    return df

# for each column in all_metric_cols, get the mean, std, etc. for each fighter, if necessary
def get_fighter_running_dist_stats(fighter, df, date, col_to_get, stat_to_calc):
    data = df[(df['Fighter_A'] == fighter) | (df['Fighter_B'] == fighter)]
    # only get fights before the date
    datey = pd.to_datetime(date)
    data['date'] = pd.to_datetime(data['date'])
    data = data[data['date'] < datey]
    # fighter could be either fighter A or fighter B
    fighter_data = pd.DataFrame()
    # when fighter is fighter A, get all fighter A data and copy it to fighter_data
    # fighterA df
    fighterA_df = data[data['Fighter_A'] == fighter]
    fighterB_df = data[data['Fighter_B'] == fighter]
    # keep only the fighters columns date, FighterA, and the col_to_get, do same for B, change col names from B to A, and concat
    fighterA_df = fighterA_df[['date', 'Fighter_A', 'A_' + col_to_get]]
    fighterA_df.rename(columns={'A_' + col_to_get: col_to_get, 'Fighter_A': 'fighter'}, inplace=True)
    fighterB_df = fighterB_df[['date', 'Fighter_B', 'B_' + col_to_get]]
    fighterB_df.rename(columns={'B_' + col_to_get: col_to_get, 'Fighter_B': 'fighter'}, inplace=True)
    fighter_data = fighter_data.append(fighterA_df)
    fighter_data = fighter_data.append(fighterB_df)
    # append the dataframes on fighter
    fighter_data = fighter_data.append(fighterA_df)
    fighter_data = fighter_data.append(fighterB_df)
    # make sure the date is before the date of the fight
    # get the average
    if stat_to_calc == 'mean':
        x = fighter_data[col_to_get].mean()
    elif stat_to_calc == 'std':
        x = fighter_data[col_to_get].std()
    elif stat_to_calc == 'max':
        x = fighter_data[col_to_get].max()
    elif stat_to_calc == 'min':
        x = fighter_data[col_to_get].min()
    elif stat_to_calc == 'median':
        x = fighter_data[col_to_get].median()  
    return x


# scrape UFCStats website for only next event to create new DF with

#st.write(next_eventz)
next_eventz['event_date'] = pd.to_datetime(next_eventz['event_date']).dt.date
d_o_e = next_eventz['event_date'].values[0]
doe = d_o_e.strftime('%Y-%m-%d')
fighter_urls = get_fighter_urls(next_eventz['event_url'].values[0])

nfd = pd.read_csv(home + 'final/next_fights/'+ doe + '_imputed.csv')

#replace na with 0
nfd.fillna(0, inplace=True)

if nfd['Fighter_A'].values[0] == next_event['fighter1'].values[0]:
    next_fight_df = nfd

else:
    next_fight_df = fighter_urls[['fighter1', 'fighter2:']]
    next_fight_df.columns = ['Fighter_A', 'Fighter_B']
    next_fight_df['date'] = d_o_e

    in_fight_cols = [n for n in list(data.columns) if n.startswith('A_') or n.startswith('B_')]
    rolling_cols = [n for n in in_fight_cols if 'Rolling' in n]
    in_fight_only_cols = [n for n in in_fight_cols if n not in rolling_cols]
    in_fight_only_cols.remove('A_Height')
    in_fight_only_cols.remove('B_Height')
    in_fight_only_cols.remove('A_Reach')
    in_fight_only_cols.remove('B_Reach')
    in_fight_only_cols.remove('A_Leg_Reach')
    in_fight_only_cols.remove('B_Leg_Reach')

    A_cols = [n for n in in_fight_only_cols if n.startswith('A_')]
    A_cols2 = pd.DataFrame(A_cols)
    A_cols2['nonspecific'] = A_cols2[0].str[2:]
    the_cols = list(A_cols2['nonspecific'].unique())

    # for each column in all_metric_cols, get the mean, std, etc. for each fighter
    def get_em(fighter, date, col_to_get, stat_to_calc):
        df = data[(data['Fighter_A'] == fighter) | (data['Fighter_B'] == fighter)]
        # only get fights before the date
        datey = pd.to_datetime(date)
        df['date'] = pd.to_datetime(data['date'])
        df = df[df['date'] < datey]
        # fighter could be either fighter A or fighter B
        fighter_data = pd.DataFrame()
        # when fighter is fighter A, get all fighter A data and copy it to fighter_data
        # fighterA df
        fighterA_df = df[df['Fighter_A'] == fighter]
        fighterB_df = df[df['Fighter_B'] == fighter]
        # keep only the fighters columns date, FighterA, and the col_to_get, do same for B, change col names from B to A, and concat
        fighterA_df = fighterA_df[['date', 'Fighter_A', 'A_' + col_to_get]]
        fighterA_df.rename(columns={'A_' + col_to_get: col_to_get, 'Fighter_A': 'fighter'}, inplace=True)
        fighterB_df = fighterB_df[['date', 'Fighter_B', 'B_' + col_to_get]]
        fighterB_df.rename(columns={'B_' + col_to_get: col_to_get, 'Fighter_B': 'fighter'}, inplace=True)
        fighter_data = fighter_data.append(fighterA_df)
        fighter_data = fighter_data.append(fighterB_df)
        # append the dataframes on fighter
        fighter_data = fighter_data.append(fighterA_df)
        fighter_data = fighter_data.append(fighterB_df)
        # make sure the date is before the date of the fight
        # get the average
        if stat_to_calc == 'mean':
            x = fighter_data[col_to_get].mean()
        elif stat_to_calc == 'std':
            x = fighter_data[col_to_get].std()
        elif stat_to_calc == 'max':
            x = fighter_data[col_to_get].max()
        elif stat_to_calc == 'min':
            x = fighter_data[col_to_get].min()
        elif stat_to_calc == 'median':
            x = fighter_data[col_to_get].median()  
        return x

    next_event_date = next_fight_df['date'].values[0]

    for col in the_cols:
        for stat in ['mean', 'std', 'max', 'min', 'median']:
            next_fight_df['A_Rolling_' + col + '_' + stat] = next_fight_df.apply(lambda x: get_em(fighter=x['Fighter_A'], date=next_event_date, col_to_get=col, stat_to_calc=stat), axis=1)
            next_fight_df['B_Rolling_' + col + '_' + stat] = next_fight_df.apply(lambda x: get_em(fighter=x['Fighter_B'], date=next_event_date, col_to_get=col, stat_to_calc=stat), axis=1)

    next_fight_df = next_fight_df.fillna(0)


this_fight_df= next_fight_df[next_fight_df['Fighter_A'] == selected_fighter_1]

###### ADD LAST DATA POINTS TO THIS FIGHT DF ######

this_fight_df['Fighter_A_Odds_obf'] = next_event['fighter1_odds'][next_event['fighter1'] == selected_fighter_1].values[0]
this_fight_df['Fighter_B_Odds_obf'] = next_event['fighter2_odds'][next_event['fighter2'] == selected_fighter_2].values[0]
this_fight_df['A_Height'] = a_height
this_fight_df['B_Height'] = b_height
this_fight_df['A_Reach'] = a_reach
this_fight_df['B_Reach'] = b_reach
this_fight_df['A_Leg_Reach'] = a_leg
this_fight_df['B_Leg_Reach'] = b_leg
this_fight_df['favorite?'] = np.where(this_fight_df['Fighter_A_Odds_obf'] < this_fight_df['Fighter_B_Odds_obf'], 1, 0)
this_fight_df['Dif_Height'] = this_fight_df['A_Height'] - this_fight_df['B_Height']
this_fight_df['Dif_Reach'] = this_fight_df['A_Reach'] - this_fight_df['B_Reach']
this_fight_df['Dif_Leg_Reach'] = this_fight_df['A_Leg_Reach'] - this_fight_df['B_Leg_Reach']



# put columns in proper order
proper_order = ['Fighter_A',
 'Fighter_B',
 'date',
 'Fighter_A_Odds_obf',
 'Fighter_B_Odds_obf',
 'A_Rolling_Kd_mean',
 'B_Rolling_Kd_mean',
 'A_Rolling_Kd_std',
 'B_Rolling_Kd_std',
 'A_Rolling_Kd_max',
 'B_Rolling_Kd_max',
 'A_Rolling_Kd_min',
 'B_Rolling_Kd_min',
 'A_Rolling_Kd_median',
 'B_Rolling_Kd_median',
 'A_Rolling_Sig_strike_land_mean',
 'B_Rolling_Sig_strike_land_mean',
 'A_Rolling_Sig_strike_land_std',
 'B_Rolling_Sig_strike_land_std',
 'A_Rolling_Sig_strike_land_max',
 'B_Rolling_Sig_strike_land_max',
 'A_Rolling_Sig_strike_land_min',
 'B_Rolling_Sig_strike_land_min',
 'A_Rolling_Sig_strike_land_median',
 'B_Rolling_Sig_strike_land_median',
 'A_Rolling_Sig_strike_att_mean',
 'B_Rolling_Sig_strike_att_mean',
 'A_Rolling_Sig_strike_att_std',
 'B_Rolling_Sig_strike_att_std',
 'A_Rolling_Sig_strike_att_max',
 'B_Rolling_Sig_strike_att_max',
 'A_Rolling_Sig_strike_att_min',
 'B_Rolling_Sig_strike_att_min',
 'A_Rolling_Sig_strike_att_median',
 'B_Rolling_Sig_strike_att_median',
 'A_Rolling_Sig_strike_percent_mean',
 'B_Rolling_Sig_strike_percent_mean',
 'A_Rolling_Sig_strike_percent_std',
 'B_Rolling_Sig_strike_percent_std',
 'A_Rolling_Sig_strike_percent_max',
 'B_Rolling_Sig_strike_percent_max',
 'A_Rolling_Sig_strike_percent_min',
 'B_Rolling_Sig_strike_percent_min',
 'A_Rolling_Sig_strike_percent_median',
 'B_Rolling_Sig_strike_percent_median',
 'A_Rolling_Total_Strikes_land_mean',
 'B_Rolling_Total_Strikes_land_mean',
 'A_Rolling_Total_Strikes_land_std',
 'B_Rolling_Total_Strikes_land_std',
 'A_Rolling_Total_Strikes_land_max',
 'B_Rolling_Total_Strikes_land_max',
 'A_Rolling_Total_Strikes_land_min',
 'B_Rolling_Total_Strikes_land_min',
 'A_Rolling_Total_Strikes_land_median',
 'B_Rolling_Total_Strikes_land_median',
 'A_Rolling_Total_Strikes_att_mean',
 'B_Rolling_Total_Strikes_att_mean',
 'A_Rolling_Total_Strikes_att_std',
 'B_Rolling_Total_Strikes_att_std',
 'A_Rolling_Total_Strikes_att_max',
 'B_Rolling_Total_Strikes_att_max',
 'A_Rolling_Total_Strikes_att_min',
 'B_Rolling_Total_Strikes_att_min',
 'A_Rolling_Total_Strikes_att_median',
 'B_Rolling_Total_Strikes_att_median',
 'A_Rolling_Total_Strikes_percent_mean',
 'B_Rolling_Total_Strikes_percent_mean',
 'A_Rolling_Total_Strikes_percent_std',
 'B_Rolling_Total_Strikes_percent_std',
 'A_Rolling_Total_Strikes_percent_max',
 'B_Rolling_Total_Strikes_percent_max',
 'A_Rolling_Total_Strikes_percent_min',
 'B_Rolling_Total_Strikes_percent_min',
 'A_Rolling_Total_Strikes_percent_median',
 'B_Rolling_Total_Strikes_percent_median',
 'A_Rolling_Takedowns_land_mean',
 'B_Rolling_Takedowns_land_mean',
 'A_Rolling_Takedowns_land_std',
 'B_Rolling_Takedowns_land_std',
 'A_Rolling_Takedowns_land_max',
 'B_Rolling_Takedowns_land_max',
 'A_Rolling_Takedowns_land_min',
 'B_Rolling_Takedowns_land_min',
 'A_Rolling_Takedowns_land_median',
 'B_Rolling_Takedowns_land_median',
 'A_Rolling_Takedowns_att_mean',
 'B_Rolling_Takedowns_att_mean',
 'A_Rolling_Takedowns_att_std',
 'B_Rolling_Takedowns_att_std',
 'A_Rolling_Takedowns_att_max',
 'B_Rolling_Takedowns_att_max',
 'A_Rolling_Takedowns_att_min',
 'B_Rolling_Takedowns_att_min',
 'A_Rolling_Takedowns_att_median',
 'B_Rolling_Takedowns_att_median',
 'A_Rolling_Takedown_percent_mean',
 'B_Rolling_Takedown_percent_mean',
 'A_Rolling_Takedown_percent_std',
 'B_Rolling_Takedown_percent_std',
 'A_Rolling_Takedown_percent_max',
 'B_Rolling_Takedown_percent_max',
 'A_Rolling_Takedown_percent_min',
 'B_Rolling_Takedown_percent_min',
 'A_Rolling_Takedown_percent_median',
 'B_Rolling_Takedown_percent_median',
 'A_Rolling_Sub_Attempts_land_mean',
 'B_Rolling_Sub_Attempts_land_mean',
 'A_Rolling_Sub_Attempts_land_std',
 'B_Rolling_Sub_Attempts_land_std',
 'A_Rolling_Sub_Attempts_land_max',
 'B_Rolling_Sub_Attempts_land_max',
 'A_Rolling_Sub_Attempts_land_min',
 'B_Rolling_Sub_Attempts_land_min',
 'A_Rolling_Sub_Attempts_land_median',
 'B_Rolling_Sub_Attempts_land_median',
 'A_Rolling_Rev_mean',
 'B_Rolling_Rev_mean',
 'A_Rolling_Rev_std',
 'B_Rolling_Rev_std',
 'A_Rolling_Rev_max',
 'B_Rolling_Rev_max',
 'A_Rolling_Rev_min',
 'B_Rolling_Rev_min',
 'A_Rolling_Rev_median',
 'B_Rolling_Rev_median',
 'A_Rolling_Ctrl_time_min_mean',
 'B_Rolling_Ctrl_time_min_mean',
 'A_Rolling_Ctrl_time_min_std',
 'B_Rolling_Ctrl_time_min_std',
 'A_Rolling_Ctrl_time_min_max',
 'B_Rolling_Ctrl_time_min_max',
 'A_Rolling_Ctrl_time_min_min',
 'B_Rolling_Ctrl_time_min_min',
 'A_Rolling_Ctrl_time_min_median',
 'B_Rolling_Ctrl_time_min_median',
 'A_Rolling_Ctrl_time_sec_mean',
 'B_Rolling_Ctrl_time_sec_mean',
 'A_Rolling_Ctrl_time_sec_std',
 'B_Rolling_Ctrl_time_sec_std',
 'A_Rolling_Ctrl_time_sec_max',
 'B_Rolling_Ctrl_time_sec_max',
 'A_Rolling_Ctrl_time_sec_min',
 'B_Rolling_Ctrl_time_sec_min',
 'A_Rolling_Ctrl_time_sec_median',
 'B_Rolling_Ctrl_time_sec_median',
 'A_Rolling_Ctrl_time_tot_mean',
 'B_Rolling_Ctrl_time_tot_mean',
 'A_Rolling_Ctrl_time_tot_std',
 'B_Rolling_Ctrl_time_tot_std',
 'A_Rolling_Ctrl_time_tot_max',
 'B_Rolling_Ctrl_time_tot_max',
 'A_Rolling_Ctrl_time_tot_min',
 'B_Rolling_Ctrl_time_tot_min',
 'A_Rolling_Ctrl_time_tot_median',
 'B_Rolling_Ctrl_time_tot_median',
 'A_Rolling_Head_Strikes_land_mean',
 'B_Rolling_Head_Strikes_land_mean',
 'A_Rolling_Head_Strikes_land_std',
 'B_Rolling_Head_Strikes_land_std',
 'A_Rolling_Head_Strikes_land_max',
 'B_Rolling_Head_Strikes_land_max',
 'A_Rolling_Head_Strikes_land_min',
 'B_Rolling_Head_Strikes_land_min',
 'A_Rolling_Head_Strikes_land_median',
 'B_Rolling_Head_Strikes_land_median',
 'A_Rolling_Head_Strikes_att_mean',
 'B_Rolling_Head_Strikes_att_mean',
 'A_Rolling_Head_Strikes_att_std',
 'B_Rolling_Head_Strikes_att_std',
 'A_Rolling_Head_Strikes_att_max',
 'B_Rolling_Head_Strikes_att_max',
 'A_Rolling_Head_Strikes_att_min',
 'B_Rolling_Head_Strikes_att_min',
 'A_Rolling_Head_Strikes_att_median',
 'B_Rolling_Head_Strikes_att_median',
 'A_Rolling_Head_Strikes_percent_mean',
 'B_Rolling_Head_Strikes_percent_mean',
 'A_Rolling_Head_Strikes_percent_std',
 'B_Rolling_Head_Strikes_percent_std',
 'A_Rolling_Head_Strikes_percent_max',
 'B_Rolling_Head_Strikes_percent_max',
 'A_Rolling_Head_Strikes_percent_min',
 'B_Rolling_Head_Strikes_percent_min',
 'A_Rolling_Head_Strikes_percent_median',
 'B_Rolling_Head_Strikes_percent_median',
 'A_Rolling_Body_Strikes_land_mean',
 'B_Rolling_Body_Strikes_land_mean',
 'A_Rolling_Body_Strikes_land_std',
 'B_Rolling_Body_Strikes_land_std',
 'A_Rolling_Body_Strikes_land_max',
 'B_Rolling_Body_Strikes_land_max',
 'A_Rolling_Body_Strikes_land_min',
 'B_Rolling_Body_Strikes_land_min',
 'A_Rolling_Body_Strikes_land_median',
 'B_Rolling_Body_Strikes_land_median',
 'A_Rolling_Body_Strikes_att_mean',
 'B_Rolling_Body_Strikes_att_mean',
 'A_Rolling_Body_Strikes_att_std',
 'B_Rolling_Body_Strikes_att_std',
 'A_Rolling_Body_Strikes_att_max',
 'B_Rolling_Body_Strikes_att_max',
 'A_Rolling_Body_Strikes_att_min',
 'B_Rolling_Body_Strikes_att_min',
 'A_Rolling_Body_Strikes_att_median',
 'B_Rolling_Body_Strikes_att_median',
 'A_Rolling_Body_Strikes_percent_mean',
 'B_Rolling_Body_Strikes_percent_mean',
 'A_Rolling_Body_Strikes_percent_std',
 'B_Rolling_Body_Strikes_percent_std',
 'A_Rolling_Body_Strikes_percent_max',
 'B_Rolling_Body_Strikes_percent_max',
 'A_Rolling_Body_Strikes_percent_min',
 'B_Rolling_Body_Strikes_percent_min',
 'A_Rolling_Body_Strikes_percent_median',
 'B_Rolling_Body_Strikes_percent_median',
 'A_Rolling_Leg_Strikes_land_mean',
 'B_Rolling_Leg_Strikes_land_mean',
 'A_Rolling_Leg_Strikes_land_std',
 'B_Rolling_Leg_Strikes_land_std',
 'A_Rolling_Leg_Strikes_land_max',
 'B_Rolling_Leg_Strikes_land_max',
 'A_Rolling_Leg_Strikes_land_min',
 'B_Rolling_Leg_Strikes_land_min',
 'A_Rolling_Leg_Strikes_land_median',
 'B_Rolling_Leg_Strikes_land_median',
 'A_Rolling_Leg_Strikes_att_mean',
 'B_Rolling_Leg_Strikes_att_mean',
 'A_Rolling_Leg_Strikes_att_std',
 'B_Rolling_Leg_Strikes_att_std',
 'A_Rolling_Leg_Strikes_att_max',
 'B_Rolling_Leg_Strikes_att_max',
 'A_Rolling_Leg_Strikes_att_min',
 'B_Rolling_Leg_Strikes_att_min',
 'A_Rolling_Leg_Strikes_att_median',
 'B_Rolling_Leg_Strikes_att_median',
 'A_Rolling_Leg_Strikes_percent_mean',
 'B_Rolling_Leg_Strikes_percent_mean',
 'A_Rolling_Leg_Strikes_percent_std',
 'B_Rolling_Leg_Strikes_percent_std',
 'A_Rolling_Leg_Strikes_percent_max',
 'B_Rolling_Leg_Strikes_percent_max',
 'A_Rolling_Leg_Strikes_percent_min',
 'B_Rolling_Leg_Strikes_percent_min',
 'A_Rolling_Leg_Strikes_percent_median',
 'B_Rolling_Leg_Strikes_percent_median',
 'A_Rolling_Distance_Strikes_land_mean',
 'B_Rolling_Distance_Strikes_land_mean',
 'A_Rolling_Distance_Strikes_land_std',
 'B_Rolling_Distance_Strikes_land_std',
 'A_Rolling_Distance_Strikes_land_max',
 'B_Rolling_Distance_Strikes_land_max',
 'A_Rolling_Distance_Strikes_land_min',
 'B_Rolling_Distance_Strikes_land_min',
 'A_Rolling_Distance_Strikes_land_median',
 'B_Rolling_Distance_Strikes_land_median',
 'A_Rolling_Distance_Strikes_att_mean',
 'B_Rolling_Distance_Strikes_att_mean',
 'A_Rolling_Distance_Strikes_att_std',
 'B_Rolling_Distance_Strikes_att_std',
 'A_Rolling_Distance_Strikes_att_max',
 'B_Rolling_Distance_Strikes_att_max',
 'A_Rolling_Distance_Strikes_att_min',
 'B_Rolling_Distance_Strikes_att_min',
 'A_Rolling_Distance_Strikes_att_median',
 'B_Rolling_Distance_Strikes_att_median',
 'A_Rolling_Distance_Strikes_percent_mean',
 'B_Rolling_Distance_Strikes_percent_mean',
 'A_Rolling_Distance_Strikes_percent_std',
 'B_Rolling_Distance_Strikes_percent_std',
 'A_Rolling_Distance_Strikes_percent_max',
 'B_Rolling_Distance_Strikes_percent_max',
 'A_Rolling_Distance_Strikes_percent_min',
 'B_Rolling_Distance_Strikes_percent_min',
 'A_Rolling_Distance_Strikes_percent_median',
 'B_Rolling_Distance_Strikes_percent_median',
 'A_Rolling_Clinch_Strikes_land_mean',
 'B_Rolling_Clinch_Strikes_land_mean',
 'A_Rolling_Clinch_Strikes_land_std',
 'B_Rolling_Clinch_Strikes_land_std',
 'A_Rolling_Clinch_Strikes_land_max',
 'B_Rolling_Clinch_Strikes_land_max',
 'A_Rolling_Clinch_Strikes_land_min',
 'B_Rolling_Clinch_Strikes_land_min',
 'A_Rolling_Clinch_Strikes_land_median',
 'B_Rolling_Clinch_Strikes_land_median',
 'A_Rolling_Clinch_Strikes_att_mean',
 'B_Rolling_Clinch_Strikes_att_mean',
 'A_Rolling_Clinch_Strikes_att_std',
 'B_Rolling_Clinch_Strikes_att_std',
 'A_Rolling_Clinch_Strikes_att_max',
 'B_Rolling_Clinch_Strikes_att_max',
 'A_Rolling_Clinch_Strikes_att_min',
 'B_Rolling_Clinch_Strikes_att_min',
 'A_Rolling_Clinch_Strikes_att_median',
 'B_Rolling_Clinch_Strikes_att_median',
 'A_Rolling_Clinch_Strikes_percent_mean',
 'B_Rolling_Clinch_Strikes_percent_mean',
 'A_Rolling_Clinch_Strikes_percent_std',
 'B_Rolling_Clinch_Strikes_percent_std',
 'A_Rolling_Clinch_Strikes_percent_max',
 'B_Rolling_Clinch_Strikes_percent_max',
 'A_Rolling_Clinch_Strikes_percent_min',
 'B_Rolling_Clinch_Strikes_percent_min',
 'A_Rolling_Clinch_Strikes_percent_median',
 'B_Rolling_Clinch_Strikes_percent_median',
 'A_Rolling_Ground_Strikes_land_mean',
 'B_Rolling_Ground_Strikes_land_mean',
 'A_Rolling_Ground_Strikes_land_std',
 'B_Rolling_Ground_Strikes_land_std',
 'A_Rolling_Ground_Strikes_land_max',
 'B_Rolling_Ground_Strikes_land_max',
 'A_Rolling_Ground_Strikes_land_min',
 'B_Rolling_Ground_Strikes_land_min',
 'A_Rolling_Ground_Strikes_land_median',
 'B_Rolling_Ground_Strikes_land_median',
 'A_Rolling_Ground_Strikes_att_mean',
 'B_Rolling_Ground_Strikes_att_mean',
 'A_Rolling_Ground_Strikes_att_std',
 'B_Rolling_Ground_Strikes_att_std',
 'A_Rolling_Ground_Strikes_att_max',
 'B_Rolling_Ground_Strikes_att_max',
 'A_Rolling_Ground_Strikes_att_min',
 'B_Rolling_Ground_Strikes_att_min',
 'A_Rolling_Ground_Strikes_att_median',
 'B_Rolling_Ground_Strikes_att_median',
 'A_Rolling_Ground_Strikes_percent_mean',
 'B_Rolling_Ground_Strikes_percent_mean',
 'A_Rolling_Ground_Strikes_percent_std',
 'B_Rolling_Ground_Strikes_percent_std',
 'A_Rolling_Ground_Strikes_percent_max',
 'B_Rolling_Ground_Strikes_percent_max',
 'A_Rolling_Ground_Strikes_percent_min',
 'B_Rolling_Ground_Strikes_percent_min',
 'A_Rolling_Ground_Strikes_percent_median',
 'B_Rolling_Ground_Strikes_percent_median',
 'A_Height',
 'B_Height',
 'Dif_Height',
 'A_Reach',
 'B_Reach',
 'Dif_Reach',
 'A_Leg_Reach',
 'B_Leg_Reach',
 'Dif_Leg_Reach',
 'favorite?']

final_vect = this_fight_df[proper_order]

st.write(final_vect)

# load model
extra_trees = pickle.load(open('C:\\Users\\tmcro\\OneDrive\\Data Science\\Personal_Projects\\Sports\\UFC_Prediction\\data\\models\\Extra_Trees_Gridsearched_7.pkl', 'rb'))

prediction = pd.DataFrame(extra_trees.predict(final_vect))
prediction = prediction[0].values[0]

if prediction == 1:
    st.sidebar.header("Predicted Winner: " + selected_fighter_1)
if prediction == 0:
    st.sidebar.header("Predicted Winner: " + selected_fighter_2)

probabilities = extra_trees.predict_proba(final_vect)
prob_win = probabilities[0][1]
prob_win = round(prob_win * 100,1)
prob_lose = probabilities[0][0]
prob_lose = round(prob_lose * 100,1)
st.sidebar.write("")
st.sidebar.subheader("Model Predicted Win Probabilities:")
st.sidebar.write(selected_fighter_1 + " : " + str(prob_win) + "%")
st.sidebar.write(selected_fighter_2 + " : " + str(prob_lose)+ "%")
st.sidebar.write("")


###########        MATCHUPS      ###############
st.sidebar.header('Selected UFC Event: '+ selected_event)

ne = next_event.rename(columns={'fighter1': 'Fighter #1', 'fighter2': 'Fighter #2', 
                                'weightclass': 'Weightclass', 'fighter1_odds': 'Fighter #1 Odds', 
                                'fighter2_odds': 'Fighter #2 Odds'})
colz = ['Fighter #1', 'Fighter #2', ]
ne = ne[colz]
st.sidebar.table(ne.style.format({'Fighter #1 Odds': '{:.2f}', 'Fighter #2 Odds': '{:.2f}'}))

st.header('Important Features')
st.write('The following features were the most important in the model')

############   ALL FEATURES  ############

cols = ['A_Rolling_Total_Strikes_land_mean', 'B_Rolling_Total_Strikes_land_mean',
        'A_Rolling_Sig_strike_land_mean', 'B_Rolling_Sig_strike_land_mean',
        'A_Rolling_Distance_Strikes_land_mean', 'B_Rolling_Distance_Strikes_land_mean', 
        'A_Rolling_Head_Strikes_land_mean', 'B_Rolling_Head_Strikes_land_mean', 'A_Rolling_Ground_Strikes_percent_median',
        'B_Rolling_Ground_Strikes_percent_median', 'A_Rolling_Ground_Strikes_percent_min', 'B_Rolling_Ground_Strikes_percent_min',
        'A_Leg_Reach', 'B_Leg_Reach', 'A_Rolling_Ctrl_time_tot_mean', 'B_Rolling_Ctrl_time_tot_mean']

a_cols = [n for n in cols if n.startswith('A')]
b_cols = [n for n in cols if n.startswith('B')]

a_cols_df = final_vect[a_cols]
b_cols_df = final_vect[b_cols]
# Make new df with a cols as rows in one column and b cols as rows in the other
df = pd.DataFrame(columns=[selected_fighter_1, selected_fighter_2])
# make column values the values from a_cols_df and b_cols_df
df[selected_fighter_1] = a_cols_df.values[0]
df[selected_fighter_2] = b_cols_df.values[0]
# round to 1
df = df.round(1)
# rows are the index of a_cols_df
df.index = a_cols
# rename indexes
df.index = ['Total Strikes (Average)', 'Total Significant Strikes (Average)', 'Distance Strikes (Average)', 'Head Strikes (Average)', 'Ground Strikes Percent (Median)',
            'Ground Strikes Percent (Minimum)', 'Leg Reach (inches)', 'Control Time (Average)']

# only display one decimal place
st.table(df.style.highlight_max(axis = 1, color = 'darkgreen').format("{:.1f}"))


st.header('Links for More Fighter Information:')
st.subheader('Wikipedia')
st.write('Follow links for fighter Wikipedia pages')

first_name1 = selected_fighter_1.split()[0]
last_name1 = selected_fighter_1.split()[1]
st.write('https://en.wikipedia.org/wiki/' + first_name1 + '_' + last_name1)

first_name2 = selected_fighter_2.split()[0]
last_name2 = selected_fighter_2.split()[1]
st.write('https://en.wikipedia.org/wiki/' + first_name2 + '_' + last_name2)

st.subheader('UFC.COM')
st.write('https://www.ufc.com/athlete/' + first_name1 + '_' + last_name1)
st.write('https://www.ufc.com/athlete/' + first_name2 + '_' + last_name2)



