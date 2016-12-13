# coding:utf-8
import gzip
import ast
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from os.path import isfile,join,isdir

import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import * 
myfont = FontProperties(fname='/usr/share/fonts/chinese/TrueType/ukai.ttf') 
matplotlib.rcParams['font.sans-serif'] = ['ukai'] #指定默认字体  
matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
from matplotlib import gridspec
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np


from argparse import ArgumentParser  
  

BASE_DIR='.'
SUFFIX='examples.txt'
PARSED_INPUT="Parsed Input: "
TRANSLATION="Translation: "
ALIGNS="[["
COVERAGE="Coverage: "

p = ArgumentParser(usage='it is usage tip', description='this is a test')  
p.add_argument('--base', default=BASE_DIR, type=str, help='base dir')  
      
args = p.parse_args()  
print args.base 



def get_div(x,y,cov,z,prefix):
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,5])
    fig=plt.figure(figsize=(20,14))
    ax1=plt.subplot(gs[1])
    ax0=plt.subplot(gs[0])

    # Draw a heatmap with the numeric values in each cell
    print cov
    bar=sns.barplot(x=cov,y=range(len(y)),orient='h',ax=ax0,palette='Blues_d')
    bar.set_yticklabels(y[::-1],rotation=0)
    bar.set_title("Coverage")
    heat=sns.heatmap(z,linewidth=.5,annot=True,fmt='d',vmin=0,vmax=100, annot_kws={"size":8},ax=ax1)
    heat.set_xticklabels(x,rotation=90,fontproperties=myfont)
    heat.set_yticklabels(y,rotation=0)
    heat.xaxis.tick_top()
    #fig=sns_plot.get_figure()
    plt.savefig(prefix+'.png')
    plt.clf()
    plt.close()


def fopen(dir_id):
    print dir_id
    filepath=join(BASE_DIR,dir_id,SUFFIX)
    if isfile(filepath):
        return open(filepath)


def reader(dir_id):
    fr=fopen(dir_id).read().split('\n')
    align_source=[]
    align_target=[]
    align_mat=[]
    align_cov=[]
    index=0
    for i in range(30):
        if fr[i].startswith(PARSED_INPUT):
            source=fr[i][len(PARSED_INPUT):].split(' ')            
        elif fr[i].startswith(TRANSLATION):
            target=fr[i][len(TRANSLATION):].split(' ')
            target=[w.decode('utf-8') for w in target]
            target.append('<eos>')       
        elif fr[i].startswith(ALIGNS):
            mat=ast.literal_eval(fr[i])
            mat=(np.asarray(mat)*100).astype(int)
            index+=1
        elif fr[i].startswith(COVERAGE):
            coverage=fr[i][len(COVERAGE):].split(' ')
            align_cov=[]
            for cov in coverage:
                covs=cov.split('/')
                if len(covs)==2:              
                    align_cov.append(float(covs[1]))
            get_div(target,source[::-1],np.asarray(align_cov),mat,str(dir_id)+'_'+str(index))



if __name__=='__main__':
    BASE_DIR=args.base
    dirs=os.listdir(BASE_DIR)
    if not os.path.isdir('graph'):
        os.mkdir('graph')
    
    os.chdir('graph')
    base=os.getcwd()
    for dir in dirs:
        if not os.path.isdir(join(BASE_DIR,dir)):
            continue
        os.chdir(base)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        os.chdir(dir)
        reader(dir)
