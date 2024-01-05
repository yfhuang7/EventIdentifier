# Import libraries
import itertools
import pandas as pd
from pandas import groupby
from pandas import Grouper
import numpy as np
import datetime as dt
import operator

import matplotlib.pyplot as plt

# Constants
cfs2cms = 0.02832  # cfs to cms
sqmi2sqkm = 2.58999  # squared mi to squred km

# Fuctions -----------------------------------
def get_streamflow(USGSids,startDT, endDT):
    """dowdload the streamflow from USGS for defined period, and resample it to hourly data

    Parameters
    ----------
    USGSids: int
        An array of the ID/Site no. of USGS gages
    startDT: str
        start date, string, "%Y-%m-%d"
    endDT: str
        end date, string, "%Y-%m-%d"

    Returns
    -------
    obs: DataFrame
        streamflow of the selected gages

    Examples
    --------
    `get_streamflow([16211600, 16213000], "2020-06-30", "2020-12-31)`
    """
    service = IVDataService()
    observations = service.get(
        sites=USGSids, 
        startDT=startDT, 
        endDT=endDT)

    # Drop extra columns to be more efficient
    obs = observations[[
        'usgs_site_code', 
        'value_time', 
        'value'
        ]]

    # Check for duplicate time series, keep first by default
    obs = obs.drop_duplicates(
        subset=['usgs_site_code', 'value_time']
        )

    # Resample to hourly, keep first measurement in each 15-min bin
    obs = obs.groupby([
        'usgs_site_code',
        Grouper(key='value_time', freq='1H')
        ]).first() ### Note, ffill will also fill those no-data.

    # convert cfs to cms
    obs = obs*cfs2cms
    return obs

def identify_highflow(df_streamflow, dir_output, gage_id, thrld_min, thrld_jump=80, thrld_flat=75, no_flat=3, thrld_duration='6H',
                  plot = True, plot_start=None, plot_end=None):
    ''' Identify high-flow events based on the streamflow/hydrograph (raising and falling limb identification) + applied thresholds.
    
    Parameters
    ----------
    df_streamflow:  pd.DataFrame
        hourly streamflow time series with `datetime` (UTC) as the index and `value` as the streamflow (cms).
    dir_output: str
        output directory.
    gage_id: str
        for creating the output file.
    thrld_min: float
        minimum peak streamflow (cms) to consider an event as identified event.
    thrld_jump: int
        percentile of changes/shifts (in an hour) to identify a "jump" (to prevent tiny jumps from unstable/sensitive streamflow). Default: 80.
    thrld_flat: int
        percentile of changes/shifts (in an hour) to identify "flat" area.
    no_flat: int
        the number of timesteps for consecutively counting as 'flat'.
    thrld_duration: str
        the threshold for the event duration. Default - at least longer than 6H.
    plot: bool 
        if you want a plot or not
    plot_start: str 
        the starting time of the plot. default is the begining of the data
    plot_end: str
        the ending time of the plot. default is the last data point.
    
    Returns
    -------
    
    '''
    # determine the "jump"
    data = df_streamflow.value-df_streamflow.value.shift(1)
    df_streamflow.loc[:,'jump'] = data>np.percentile([x for x in data if x>0], thrld_jump)  # need to have this to prevent getting tiny jump from unstable streamflow

    ## Looking for the "flat" area (the first of flat should be the end of falling limb): negative changes, streamflow varies within threld_flat percentile of change/shifts
    df_streamflow.loc[:, 'flat'] = (data<=0.01)&(abs(data)<=(np.percentile([abs(x) for x in data if x<0], thrld_flat)))

    ### Extract the indices of jump and flat
    # indices for 'jumps'
    count_dups = [[i, sum(1 for i in group)] for i, group in groupby(df_streamflow['jump'])]
    results = itertools.accumulate(list(map(operator.itemgetter(1), count_dups)), operator.add)
    idx = [each for each in results]

    # Add the accumulated index 
    for i in range(len(count_dups)):
        count_dups[i].append(idx[i])

    idx_jump = [idx[2]-idx[1] for idx in count_dups if (idx[0]==True)]

    # indices for the 'flats'
    count_dups = [[i, sum(1 for i in group)] for i, group in groupby(df_streamflow['flat'])]
    results = itertools.accumulate(list(map(operator.itemgetter(1), count_dups)), operator.add)
    idx = [each for each in results]

    # Add the accumulated index 
    for i in range(len(count_dups)):
        count_dups[i].append(idx[i])

    # flat and decay. both needs to have at least three consecutive flat/deay
    idx_flat_start = [idx[2]-idx[1]-1 for idx in count_dups if (idx[0]==True)&(idx[1]>=no_flat)]  #start of flat
    idx_flat_end = [idx[2]-1 for idx in count_dups if (idx[0]==True)&(idx[1]>=no_flat)]   # end of flat

    df_streamflow.loc[:,'flat'] = False
    df_streamflow.loc[:,'flat'].iloc[idx_flat_start]=True

    ### List the event
    from datetime import datetime
    start = []
    end = []

    ## Found the first timestep of "flat" after a jump
    for i in range(len(idx_jump)):
        start_tmp = df_streamflow.iloc[idx_jump].index[i]
        start.append(min(df_streamflow.iloc[idx_flat_end].index, key=lambda x: (x>start_tmp, abs(start_tmp-x))))
        end.append(min(df_streamflow.iloc[idx_flat_start].index, key=lambda x: (x<start_tmp, abs(x-start_tmp))))

    ## Create an empty list for storing event info
    event_list = pd.DataFrame({
                    'start': start,
                    'end': end,
                    'Qmax': np.nan,
                    'Qmin': np.nan,
                    'Qrange': np.nan,
                    'Qmean': np.nan,
                    'Qmedian': np.nan,
                    'Qstd': np.nan,
                    'Qvar': np.nan,
                    'Qskew': np.nan,
                    'Qkurt': np.nan,
                    'Qtotal': np.nan,
                    'Qnpeak': np.nan,
                    'Qlen': np.nan,
                    'Qtrange': np.nan,
                 })
    
    ## Drop the event that is within another event
    event_list = event_list.drop_duplicates(subset='end',keep="first")
    event_list = event_list.drop_duplicates(subset='start',keep="last")

    ## Extend the event starting/ending time for 3 hour, respectively
    event_list.loc[:,'start'] = event_list['start'] - dt.timedelta(hours=3)
    event_list.loc[:,'end'] = event_list['end'] + dt.timedelta(hours=3)

    ### Discard the event that is shorter than thrld_duration
    event_list = event_list[(event_list.end-event_list.start)>=thrld_duration]

    ## Mark the event_list back to the df_streamflow
    df_streamflow.loc[:,'event'] = False

    for i in range(len(event_list)):
        start = event_list.start.iloc[i]
        stop = event_list.end.iloc[i]
        Qevent = df_streamflow.loc[start:stop]
        ### filter out if the event max. larger than the threshold
        if Qevent.max().value>=thrld_min:

            # Calculate event statistics
            ## magnitude related
            Qmax = Qevent.value.max()    #cms, maximum Q
            Qmin = Qevent.value.min()    #cms, minmum Q
            Qrange = Qmax - Qmin    #cms, range of Q
            Qmean = Qevent.value.mean()    # cms, mean Q
            Qmedian = Qevent.value.median()    # cms, median Q
            Qstd = Qevent.value.std()    # cms, standard deviation
            Qvar = Qevent.value.var()    # cms, variance
            Qskew = Qevent.value.skew()    # skewness
            Qkurt = Qevent.value.kurt()    # kurt index

            Qtotal = Qevent.value.sum()*3600  #m^3, total event discharge
            Qnpeak = Qevent.jump.sum() #the number of peaks in the event

            ## time related
            Qlen = len(Qevent) # event duration (time interval must be "hour" here), event length
            Qtrange = int((Qevent.loc[Qevent.value==Qmax,'value'].index - start).total_seconds()[0]/3600) - 3 # hour, time to peak (from first Q rise, not starts of event)

            ## mark the event
            df_streamflow.loc[start:stop,'event'] = True

            ## write the statistics to the event_list
            event_list.iloc[i] = [start, stop, Qmax, Qmin, Qrange, Qmean, Qmedian, Qstd, Qvar, Qskew, Qkurt, Qtotal, Qnpeak, Qlen, Qtrange]

    
    # Tidy up the dataframes
    event_list.dropna(inplace=True)
    df_streamflow.drop(['jump','flat'], axis=1, inplace=True)
    df_streamflow.columns = ['Q_cms', 'event']
    df_streamflow.index.name = 'Datetime'

    event_list.to_csv(dir_output + 'event_list_' + gage_id + '.csv', index=False)
    df_streamflow.to_csv(dir_output + 'obs_event_marked_' + gage_id + '.csv', index=True)

    ## Plot
    if plot==True:
        if plot_start is None:
            plot_start = np.nanmin(df_streamflow.index)
        if plot_end is None:
            plot_end = np.nanmax(df_streamflow.index)

        df_streamflow_trim = df_streamflow.loc[plot_start:plot_end]

        df_streamflow_trim.loc[df_streamflow_trim.event==False,'event'] = 'blue'
        df_streamflow_trim.loc[df_streamflow_trim.event==True, 'event'] = 'orange'

        plt.subplots(figsize=(20,8))
        plt.scatter(df_streamflow_trim.reset_index()['Datetime'], 
                    y=df_streamflow_trim.reset_index()['Q_cms'], 
                    c=df_streamflow_trim.reset_index()['event'], s=10)

        plt.ylabel('Discharge (cms)', labelpad=10, fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.savefig(dir_output + 'plot_events_' + gage_id + '.jpg', dpi=300)
    plt.close()
