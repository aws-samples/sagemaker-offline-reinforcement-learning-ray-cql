TUNING_JOB_NAME = "offline-rl-1000-iter-230505-1756"
# TUNING_JOB_NAME = "offline-rl-1000-iter-230504-2327"
COLUMNS_TO_PLOT = ['critic_loss', 'evaluation_critic_loss',  'td_mse', 'evaluation_td_mse', 'cql_loss', 'evaluation_cql_loss', 'actor_loss','evaluation_actor_loss','training_iteration','iterations_since_restore', 'timesteps_total', 'mean_q']
# COLUMNS_TO_PLOT = ['critic_loss', 'evaluation_avg_critic_loss',  'td_mse', 'evaluation_td_mse', 'actor_loss','evaluation_avg_actor_loss','training_iteration','iterations_since_restore', 'timesteps_total', 'mean_q','cql_loss']

NUM_COLUMNS = 2

import os
import sagemaker
import boto3
from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# init_notebook_mode(connected=True)         # initiate notebook for offline plot

pio.templates.default = "simple_white"
pd.options.plotting.backend = "plotly"

sm_client = boto3.client('sagemaker')
cw_client = boto3.client('cloudwatch')


num_rows = math.ceil((1+len(COLUMNS_TO_PLOT))/float(NUM_COLUMNS))

training_job_names = sm_client.list_training_jobs_for_hyper_parameter_tuning_job(HyperParameterTuningJobName=TUNING_JOB_NAME, MaxResults = 100)
print(f'{len(training_job_names["TrainingJobSummaries"])} training jobs')
metric_series = {}
for job in training_job_names['TrainingJobSummaries']:
    # print(job)
    available_metrics = cw_client.list_metrics(
        Namespace="/aws/sagemaker/TrainingJobs",
        # MetricName='string',
        Dimensions=[
            {
                'Name': 'TrainingJobName',
                'Value': job['TrainingJobName']
            },
        ],
    )
    MetricDataQueries=[ {
            'Id': metric['MetricName'].lower(),
            'MetricStat': {
                'Metric': metric,
                'Stat': 'Minimum', #Average
                'Period': 300
             }
        } for metric in available_metrics['Metrics']]
    
    
    # print(f'MetricDataQueries: {MetricDataQueries}')
    
    cw_metric_data = cw_client.get_metric_data(
        MetricDataQueries=MetricDataQueries,
        StartTime=job['TrainingStartTime'],
        EndTime=job.get('TrainingEndTime',datetime.now()),
    )
    
    
    # dict_of_series = {metric_data['Label']: pd.Series(metric_data['Values'], index = metric_data['Timestamps'], name = job['TrainingJobName']) for metric_data in cw_metric_data['MetricDataResults']}
    dict_of_series = {metric_data['Label']: pd.Series(metric_data['Values'][::-1], name = job['TrainingJobName']) for metric_data in cw_metric_data['MetricDataResults']}
    
    
    
    for key, value in dict_of_series.items():
        # print(key)
        # print(value)
        metric_series[key] = metric_series.get(key,[]) + [value]

specs = [[{"type": "table", "rowspan": num_rows}]+[{"type": "scatter"}]*(NUM_COLUMNS)]+[ [None] + [{"type": "scatter"}]*NUM_COLUMNS]*(num_rows-1)

# print(f'specs: {specs}')


fig = make_subplots(
    rows=num_rows, 
    cols=NUM_COLUMNS+1, 
    subplot_titles = ['Latest Values'] + COLUMNS_TO_PLOT, #'Last Value Table'
    specs=specs
)
fig['layout'].update(height=1000, width=1500, title_text=f"Tuning Job: {TUNING_JOB_NAME}")

plot_locs = list(range(1,1+len(COLUMNS_TO_PLOT)))
plot_locs = [ (i+1,j+2) for i in range(num_rows) for j in range(NUM_COLUMNS)]

# colors = px.colors.qualitative.Vivid
colors = px.colors.qualitative.Alphabet

latest_value_table = pd.DataFrame()

# Plot the trends
for plot_loc, column_name in zip(plot_locs,COLUMNS_TO_PLOT):
    if column_name not in metric_series: 
        print(f'{column_name} not in metric series')
        continue
    
    df = pd.concat(metric_series[column_name], axis = 1)
    
    latest_value_table[column_name] = pd.concat([series.iloc[-1:].reset_index(drop = True) for series in metric_series[column_name]], axis = 1).transpose()
    
    
    for trial in df.columns:
        i = trial.split('-')[-2]
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[trial],
                mode='lines', # 'lines' or 'markers'
                name= f'{i}. {trial}',
                legendgroup=trial,
                # line = {'color': colors[hash(trial)%len(colors)]},
                line = {'color': colors[int(i)%len(colors)]},
                showlegend = plot_loc == plot_locs[1]
            ),
        row=plot_loc[0], 
        col=plot_loc[1])
        
        # Use a log scale if all values are above 0.
        if df[trial].min() > 0:
            fig.update_yaxes(
                type="log", 
                row=plot_loc[0], 
                col=plot_loc[1])

        # fig['layout'].update(height=1000, width=1500) 
    

latest_value_table['Trial #'] = latest_value_table.index.to_series().apply(lambda x: x.split('-')[-2])
latest_value_table = latest_value_table.round(2).reset_index(drop = True)[['Trial #','td_mse','cql_loss','actor_loss','mean_q','training_iteration']]
latest_value_table.rename(columns={'training_iteration':'iters'}, inplace = True)
latest_value_table.sort_values('mean_q', ascending = False, inplace = True)
# df_summary_table = pd.concat(metric_series[SUMMARY_TABLE_COLUMN], axis = 1).describe().transpose()[['mean','std','count']].round(2)
# # df_summary_table.drop(index=True,inplace=True)
# df_summary_table.reset_index(drop=True, inplace=True)
# df_summary_table.reset_index(inplace=True)
# df_summary_table.sort_values('mean', ascending = False, inplace = True)
# Plot a summary table
fig.add_trace(
    go.Table(
        header={'values': [col.replace("_"," ") for col in latest_value_table.columns]},
        cells=dict(
            values=[latest_value_table[k].tolist() for k in latest_value_table.columns],
            # align = "left",
            # width=50
            )
    ),
    row=1, 
    col=1)
                     
# fig.show()
fig.write_html(os.path.join(os.getcwd(),f'{TUNING_JOB_NAME}_hyperparameter_tuning_plot.html'))