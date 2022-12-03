# helper.py
import matplotlib.pyplot as plt
import numpy as np

#Print unique values
def print_unique(df):
    for col in df.columns:
        print('\n'+col+'\n', df[col].unique())
        
#Remove empty values
def remove_empties(dataframe, skip_col):
    for col in dataframe.columns:
        if col==skip_col:
            continue 
        dataframe = dataframe[dataframe[col].notnull()]
    return dataframe

# colum: name of the column where to look for the value
# value: value that we want to standardize
# wrong_value_list: list of alternative values that we want to replace with 'value'
def fix_column(df,column,value,wrong_value_list):
    for i in df.index:
        if df.at[i, column] in wrong_value_list:
            df.at[i, column] = value
            
def print_selection_coord(df,ppm_min,name):
    df=df[['sample','easting_wgs84','northing_wgs84','geometry']]
    df.to_csv('data/'+name+'_'+str(ppm_min)+'ppm.csv')
    

def plot_geo(df1_1,df1_2, df2_1, df2_2, max_ppm1, max_ppm2, color1, color2, label1, label2,title1,title2,plotname):
    # binning. avoid different x,y ranges to have better comparison.
    binwidth = 25.
    max_entries = 120.
    min_ppm = 0.
    max_ppm = max( max_ppm1, max_ppm2)
    binning_ppm = (np.arange(min_ppm, max_ppm + binwidth, binwidth))

    #aestetics
    color_basalt = 'purple'
    color_gabbro = 'green'
    color_ni = 'blue'
    color_co = 'red'
    line_width = 1.5
    hist_type = 'step'

    # plot by metal
    fig = plt.figure(figsize=(12, 6))
    ax  = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax.hist(df1_1, bins=binning_ppm, density=False, histtype=hist_type, linewidth=line_width, color=color1, label=label1)
    ax.hist(df1_2, bins=binning_ppm, density=False, histtype=hist_type, linewidth=line_width, color=color2, label=label2)
    ax.set_title(title1)
    ax.set_ylabel('Deposits')
    ax.set_xlabel('ppm')
    ax.set_ylim([0, max_entries])

    ax2.hist(df2_1, bins=binning_ppm, density=False, histtype=hist_type, linewidth=line_width, color=color1, label=label1)
    ax2.hist(df2_2, bins=binning_ppm, density=False, histtype=hist_type, linewidth=line_width, color=color2, label=label2)
    ax2.set_title(title2)
    ax2.set_ylabel('Deposits')
    ax2.set_xlabel('ppm')
    ax2.set_ylim([0, max_entries])

    leg2 = ax2.legend(loc='upper right', fontsize='medium')
    plt.savefig('plots/'+plotname+'.png')
    plt.show()