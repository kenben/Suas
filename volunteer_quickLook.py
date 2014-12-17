# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from __future__ import print_function
from statsmodels.compat import lmap
from scipy import stats
import statsmodels.api as sm

import matplotlib
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 12}

#%matplotlib inline
matplotlib.rc('font', **font)

# <codecell>

cd ~/Dropbox/DataDive/volunteer_programme/2013/

# <markdowncell>

# ### Let's load some data

# <codecell>

all_apps = pd.read_table('2013_01_applicant_personalInformation.csv',sep=',')
all_apps.columns = [re.sub(r'\W+', '_', x) for x in all_apps.columns]
# need to fix some datatypes
all_apps['Age_at_end_of_2012'] = all_apps['Age_at_end_of_2012'].astype(np.float64)

# load the ID to school
plac = pd.read_table('2013_06_volunteer_placements.csv',sep=',')
plac.columns = [re.sub(r'\W+', '_', x) for x in plac.columns]

# load the interview assessments
ass = pd.read_table('2013_04_applicant_interview_assessment.csv',sep=',')
ass.columns = [re.sub(r'\W+', '_', x) for x in ass.columns]
ass['Interview_Suggested_Ranking_numerical_'] = ass['Interview_Suggested_Ranking_numerical_'].astype(np.float64)
ass['Character_1_very_introvert_5_very_extrovert_'] = ass['Character_1_very_introvert_5_very_extrovert_'].astype(np.float64)
ass['interview_outcome_1_no_3_unsure_5_yes_'] = ass['interview_outcome_1_no_3_unsure_5_yes_'].astype(np.float64)

# load the other stuff

ques = pd.read_table('2013_02_applicant_questions.csv',sep=',')
ques.columns = [re.sub(r'\W+', '_', x) for x in ques.columns]

att = pd.read_table('2013_03_applicant_interview_offer_and_attendance.csv',sep=',')
att.columns = [re.sub(r'\W+', '_', x) for x in att.columns]

rank = pd.read_table('2013_02b_applicant_rankings.csv',sep=',')
rank.columns = [re.sub(r'\W+', '_', x) for x in rank.columns]
rank['SkillsAndExperience_Music'] = rank['SkillsAndExperience_Music'].astype(np.float64)

# Ok, there is a bunch of values missing, and that should be 0, not NaN. Let's change them:
rank.loc[pd.isnull(rank.loc[:,'SkillsAndExperience_Sport']),'SkillsAndExperience_Sport'] = 0
rank.loc[pd.isnull(rank.loc[:,'SkillsAndExperience_Music']),'SkillsAndExperience_Music'] = 0
rank.loc[pd.isnull(rank.loc[:,'SkillsAndExperience_Drama']),'SkillsAndExperience_Drama'] = 0
rank.loc[pd.isnull(rank.loc[:,'SkillsAndExperience_Art']),'SkillsAndExperience_Art'] = 0
rank.loc[pd.isnull(rank.loc[:,'SkillsAndExperience_TeachOrCoach']),'SkillsAndExperience_TeachOrCoach'] = 0

# <markdowncell>

# ### Merging everything on volunteer ID
# 
# - We want one file that lists everything by volunteer.
# - Note: There are two volunteers who go to India, but didn't apply first (they seem to have been deferred a previous admission). Thus, use outer joins!

# <codecell>

# left join all tables on Person_ID
v_merged = pd.merge(left=all_apps, right=plac, how='outer', on='Person_ID')
v_merged = pd.merge(left=v_merged,right=ass, how='outer',on='Person_ID')
v_merged = pd.merge(left=v_merged,right=ques, how='outer',on='Person_ID')
v_merged = pd.merge(left=v_merged,right=rank, how='outer',on='Person_ID') 
v_merged = pd.merge(left=v_merged,right=att, how='outer',on='Person_ID') 

#v_merged.loc[v_merged.Person_ID == 170,:]
#v_merged.loc[:,['Person_ID','Interview_Suggested_Ranking_numerical_', 'Centre_ID']]

# <markdowncell>

# ### What does the application funnel look like?

# <markdowncell>

# #### How many people go through to the interview stage?

# <codecell>

vg = v_merged.groupby('Interview_offered')
vg.size()

# <markdowncell>

# #### How many people accepted the interview offer?

# <codecell>

vg = v_merged.groupby('Accepted_Interview_offer')
vg.size()

# <markdowncell>

# #### How many people came to their interview?

# <codecell>

v_merged.columns
vg = v_merged.groupby('Attended_Interview')
vg.size()

# <markdowncell>

# #### How many people received offers after their interview?

# <codecell>

vg = v_merged.groupby('Place_offered_offer_holding_rejected_')
b = vg.size()
print('Accepted: ', b.sum() - b.loc['holding'] - b.loc['rejected'])
print('Rejected:', b.loc['rejected'])
print('Waitlist: ', b.loc['holding'])

# <markdowncell>

# #### How many people go to India?

# <codecell>

print('Volunteers who go: ', (pd.isnull(v_merged['Centre_ID']) == False).sum())

# <markdowncell>

# ## Now we bring in the student improvements
# 
# Every student has several volunteers who interact with them. We want all combinations - so if there are two volunteers A and B who interact with students X and Y, then there would be four points: (A,X), (A,Y), (B,X), and (B,Y).

# <codecell>

# read the childrens' data
st = pd.read_table('child_assessments/2013_Child_Learning Assessment_Volunteer_Centres.csv',sep=',')

# fix the column names to be simpler to handle
st.columns = [re.sub(r'\W+', '_', x) for x in st.columns]

# fix some data types
st['Age'] = st['Age'].convert_objects(convert_numeric=True)
st['PostTest_Literacy'] =  st.PostTest_Literacy.astype('int64')
st['PreTest_Literacy'] = st['PreTest_Literacy'].astype(np.float64)
st['PreTest_Literacy'] = st['PreTest_Literacy'].astype(np.float64)
st['PreTest_Words'] = st['PreTest_Words'].astype(np.float64)
st['PostTest_Words'] = st['PostTest_Words'].astype(np.float64)
st['PreTest_Letters'] = st['PreTest_Letters'].astype(np.float64)
st['PostTest_Letters'] = st['PostTest_Letters'].astype(np.float64)
st['PreTest_Confidence'] = st['PreTest_Confidence'].astype(np.float64)
st['PostTest_Confidence'] = st['PostTest_Confidence'].astype(np.float64)

# now let's calculate post-pre improvements of the children
st['word_diff'] = st.PostTest_Words - st.PreTest_Words
st['letter_diff'] = st.PostTest_Letters - st.PreTest_Letters
st['lit_diff'] = st.PostTest_Literacy - st.PreTest_Literacy
st['conf_diff'] = st.PostTest_Confidence - st.PreTest_Confidence
st['norm_diff'] = (st.PostTest_Words / 5. + st.PostTest_Letters - 
                                    st.PreTest_Words / 5. - st.PreTest_Letters)

# inner join on student by centre ID, thus creating (students) x (volunteer) many entries
st_v_merged = pd.merge(left=st,right=v_merged,on='Centre_ID', how='inner')

st_v_merged.head()

# <markdowncell>

# ## Do students improve significantly?
# 
# These are paired tests. Let's check the p-values:

# <codecell>

mdiff = (st.lit_diff).mean()
(t,p) = stats.ttest_rel(st.PostTest_Literacy,st.PreTest_Literacy)

print('Avg. change in literacy: %.02f'%mdiff)
print('p-value: %.04f'%p)

mdiff = (st.letter_diff).mean()
(t,p) = stats.ttest_rel(st.PostTest_Letters,st.PreTest_Letters)

print('\nAvg. change in letters: %.02f'%mdiff)
print('p-value: %.04f'%p)

mdiff = (st.word_diff).mean()
(t,p) = stats.ttest_rel(st.PostTest_Words,st.PreTest_Words)

print('\nAvg. change in words: %.02f'%mdiff)
print('p-value: %.04f'%p)

idx = (False == pd.isnull(st.conf_diff))
mdiff = (st.PostTest_Confidence[idx] - st.PreTest_Confidence[idx]).mean()
(t,p) = stats.ttest_rel(st.PreTest_Confidence[idx], st.PostTest_Confidence[idx])

print('\nAvg. change in confidence: %.02f'%mdiff)
print('p-value: %.04f'%p)

# <markdowncell>

# Those are pretty strong values. We should have a look at the data to make sure it's not just outliers.

# <codecell>

cns = [('PreTest_Confidence','PostTest_Confidence'), ('PreTest_Literacy','PostTest_Literacy'), 
       ('PreTest_Letters','PostTest_Letters'), ('PreTest_Words','PostTest_Words')]
cats = ['Confidence','Literacy','Letters','Words']

fig = plt.figure()
for n,(cn,c) in enumerate(zip(cns,cats)):
    ax = fig.add_subplot(2,2,n)
    ax.scatter(st[cn[0]],st[cn[1]])
    ax.set_xlabel('pre')
    ax.set_ylabel('post')
    ax.set_title(c)

fig.tight_layout()
plt.show()   

# <markdowncell>

# That looks pretty ok in terms of outliers, and convincing in terms of improvements. It seems the students did get better.
# 
# ## Check against control centres
# 
# However, maybe this is not due to the volunteers, but rather because the students just had more classes (or got to know the test). So let's compare this against the data from the 'control centres', where there were _no_ Suas volunteers.

# <codecell>

# load the data from the control centres
dtys = {'PreTest_Literacy': np.float64,'PostTest_Literacy': np.float64,
        'PreTest_Words': np.float64, 'PostTest_Words': np.float64, 'PreTest_Confidence': np.float64,
        'PostTest_Confidence': np.float64, 'PreTest_Letters': np.float64, 'PostTest_Letters': np.float64,
        'Age': np.float64}
cs = pd.read_table('child_assessments/2013_Child_Learning Assessment_Control_Centres.csv', sep=',',dtype=dtys)

# fix the column names to be simpler to handle
cs.columns = [re.sub(r'\W+', '_', x) for x in cs.columns]

# now let's calculate post-pre improvements of the children
cs['word_diff'] = cs.PostTest_Words - cs.PreTest_Words
cs['letter_diff'] = cs.PostTest_Letters - cs.PreTest_Letters
cs['lit_diff'] = cs.PostTest_Literacy - cs.PreTest_Literacy
cs['conf_diff'] = cs.PostTest_Confidence - cs.PreTest_Confidence
cs['norm_diff'] = (cs.PostTest_Words / 5. + cs.PostTest_Letters - 
                                    cs.PreTest_Words / 5. - cs.PreTest_Letters)

mdiff = (cs.PostTest_Literacy - cs.PreTest_Literacy).mean()
(t,p) = stats.ttest_rel(cs.PostTest_Literacy,cs.PreTest_Literacy)

print('Avg. change in literacy: %.02f'%mdiff)
print('p-value: %.04f'%p)

mdiff = (cs.PostTest_Letters - cs.PreTest_Letters).mean()
(t,p) = stats.ttest_rel(cs.PostTest_Letters,cs.PreTest_Letters)

print('\nAvg. change in letters: %.02f'%mdiff)
print('p-value: %.04f'%p)

mdiff = (cs.PostTest_Words - cs.PreTest_Words).mean()
(t,p) = stats.ttest_rel(cs.PostTest_Words,cs.PreTest_Words)

print('\nAvg. change in words: %.02f'%mdiff)
print('p-value: %.04f'%p)

idx = (False == pd.isnull(cs.PreTest_Confidence - cs.PostTest_Confidence))
mdiff = (cs.PostTest_Confidence[idx] - cs.PreTest_Confidence[idx]).mean()
(t,p) = stats.ttest_rel(cs.PreTest_Confidence[idx], cs.PostTest_Confidence[idx])

print('\nAvg. change in confidence: %.02f'%mdiff)
print('p-value: %.04f'%p)

# plot the same columns as in the 'volunteer centre' dataset
fig = plt.figure()
for n,(cn,c) in enumerate(zip(cns,cats)):
    ax = fig.add_subplot(2,2,n)
    ax.scatter(cs[cn[0]],cs[cn[1]])
    ax.set_xlabel('pre')
    ax.set_ylabel('post')
    ax.set_title(c)

fig.tight_layout()
plt.show()   

# <markdowncell>

# That's great - the children in the control centres also improved significantly over the summer. Maybe, though, the children in the volunteer centres improved significantly _more_ than the children in the control centres?
# 
# ### Differences of paired scores

# <codecell>

cohens_d = lambda c0,c1: (c0.mean() - c1.mean()) / (np.sqrt((c0.std() ** 2 + c1.std() ** 2) / 2))
(t,p) = stats.ttest_ind(st.letter_diff, cs.letter_diff)
print('Comparison of change in letters')
print('Effect size: %.02f'%cohens_d(st.letter_diff,cs.letter_diff))
print('t: %.02f'%t)
print('p-value: %.04f'%p)

(t,p) = stats.ttest_ind(st.word_diff, cs.word_diff)
print('\nComparison of change in words')
print('Effect size: %.02f'%cohens_d(st.word_diff, cs.word_diff))
print('t: %.02f'%t)
print('p-value: %.04f'%p)

(t,p) = stats.ttest_ind(st.lit_diff, cs.lit_diff)
print('\nComparison of change in literacy')
print('Effect size: %.02f'%cohens_d(st.lit_diff, cs.lit_diff))
print('t: %.02f'%t)
print('p-value: %.04f'%p)

(t,p) = stats.ttest_ind((st.conf_diff).dropna(), 
                        (cs.conf_diff).dropna())
print('\nComparison of change in confidence')
print('Effect size: %.02f'%cohens_d((st.conf_diff).dropna(), 
                        (cs.conf_diff).dropna()))
print('t: %.02f'%t)
print('p-value: %.04f'%p)

fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(231)
sm.qqplot((st.word_diff), line='s',ax=ax1)
ax1.set_title('volunteer centres, change in word scores')
ax2 = fig.add_subplot(234)
sm.qqplot(cs.word_diff, line='s',ax=ax2) 
ax2.set_title('control centres, change in word scores')

ax3 = fig.add_subplot(232)
sm.qqplot((st.letter_diff), line='s',ax=ax3)
ax3.set_title('volunteer centres, change in letter scores')
ax4 = fig.add_subplot(235)
sm.qqplot(cs.letter_diff, line='s',ax=ax4) 
ax4.set_title('control centres, change in word scores')

ax5 = fig.add_subplot(233)
sm.qqplot((st.conf_diff.dropna()), line='s',ax=ax5)
ax5.set_title('volunteer centres, change in confidence scores')
ax6 = fig.add_subplot(236)
sm.qqplot(cs.conf_diff.dropna(), line='s',ax=ax6) 
ax6.set_title('control centres, change in confidence scores')

plt.tight_layout()
plt.show()

# <markdowncell>

# #### Permutation Tests
# The qq-plots aren't great. The data is definitely heavy-tailed, and there are some outliers. Maybe there is some unaccounted effect in the data?
# 
# For now, let's do a quick permutation test, to cross-check the results from the t-test. (This takes a short moment to run.)

# <codecell>

# stackoverflow to the rescue...
def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0.
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

print('Permutation test for letters: ',exact_mc_perm_test(st.letter_diff,cs.letter_diff,100000))
print('Permutation test for words: ',exact_mc_perm_test(st.word_diff,cs.word_diff,100000))
print('Permutation test for confidence: ',exact_mc_perm_test(st.conf_diff.dropna(),cs.conf_diff.dropna(),100000))

# <markdowncell>

# Great - that matches the t-test results fairly closely.
# 
# One more test, for the word scores: let's remove those extreme values and repeat the test.

# <codecell>

lb,up = st.word_diff.describe(percentiles=[0.1,0.9]).loc[['10%','90%']]
g1 = st.word_diff.ix[(st.word_diff > lb) & (st.word_diff < up)]
lb,up = cs.word_diff.describe(percentiles=[0.1,0.9]).loc[['10%','90%']]                                                         
g0 = cs.word_diff.ix[(cs.word_diff > lb) & (cs.word_diff < up)]
#sm.qqplot(g1, line='s')
#sm.qqplot(g0, line='s')
exact_mc_perm_test(g0, g1, 100000)

# <markdowncell>

# Alright, that looks good. It's convincing that for 'words', the children with volunteers improved significantly more than those in the control centres. Time to move on!

# <markdowncell>

# ### Which children improve how much?
# 
# Let's sort out which students improve the most. Are there schools where the children learn more? Or maybe children of some age learn the most?

# <codecell>

fig1 = plt.figure('Volunteer Centres')
fig2 = plt.figure('Control Centres')
pd.scatter_matrix(st.ix[:,['Age','word_diff']],ax=fig1.add_subplot(111))
pd.scatter_matrix(cs.ix[:,['Age','word_diff']],ax=fig2.add_subplot(111))
plt.show()

print('Volunteer Centres')
print(st.ix[:,['Age','word_diff']].corr())
print('\n')
print('Control Centres')
print(cs.ix[:,['Age','word_diff']].corr())

# <markdowncell>

# Meh. Not too much to see there. The inverse 'u' shape in the control centres could be interesting - looks like the very old and very young kids don't learn so much? Let's look at influence of **school** next.

# <codecell>

stg = st.groupby('Centre_ID')
cs.groupby('Organisation')

st.boxplot(column=['word_diff'],by='Centre_ID',figsize=(15,5))
st.boxplot(column=['letter_diff'],by='Centre_ID',figsize=(15,5))

st.boxplot(column=['Age'],by='Centre_ID',figsize=(15,5))
plt.show()

# <markdowncell>

# Ok, that starts to look dramatic. But the sample sizes are pretty small. Time for a test! Let's do a one-way ANOVA, by school (centre).

# <codecell>

from statsmodels.formula.api import ols

clean_st = st.dropna(subset=['word_diff','Centre_ID','Age'])

lm = ols('word_diff ~ Age + C(Centre_ID)',
         data=clean_st).fit()

table = sm.stats.anova_lm(lm, typ=3)
print(table)

print(lm.summary())

# <markdowncell>

# Need to do:
# - Plots of residuals vs fitted
# - QQ-plot of residuals

# <markdowncell>

# Keep in mind that there are really not a lot of students per school:

# <codecell>

st.groupby('Centre_ID').size()

# <codecell>

fix, ax = plt.subplots(figsize=(12,14))
#fig = sm.graphics.plot_partregress("Age", "word_diff", ["Centre_ID"], data=st, ax=ax)

sm.graphics.plot_partregress(endog='word_diff', exog_i='Age', exog_others=['Centre_ID'], 
        data=clean_st, ax=ax)
plt.show()

subset = ~clean_st.index.isin([68])
lm2 = ols('word_diff ~ Age + C(Centre_ID)',
         data=clean_st,subset=subset).fit()

print(lm2.summary())

# <markdowncell>

# # Snippets. Might come back to this later:

# <codecell>

from scipy.stats import pearsonr
from sklearn.covariance import MinCovDet

# just look at what's interesting for now, and drop the NAs involved
clean = st_v_merged.loc[:,['norm_diff','Interview_Suggested_Ranking_numerical_']]
clean = clean.dropna(axis=0)

# calculate robust covariance estimate, calculate what's too far away
mcd = MinCovDet()
mcd.fit(clean)

pearsonr(clean.iloc[:,0],clean.iloc[:,1])

# <codecell>

d = mcd.mahalanobis(clean)
d.sort()
d

