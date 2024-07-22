import pandas as pd
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings # Suppress warnings 
warnings.filterwarnings('ignore')
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.subplots as sp

np.random.seed(7)

def box_plot(data):
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=data['data_0'],
        name='data_0',
        marker_color='blue',
        opacity=0.7
    ))
    fig.add_trace(go.Box(
        y=data['data_1'],
        name='data_1',
        marker_color='orange',
        opacity=0.7
    ))

    fig.update_layout(
        title='Box Plot of Data_0 and Data_1',
        title_x=0.5,
        title_font=dict(size=20, family='Arial, sans-serif'),
        template='plotly_white',
        font=dict(size=14),
        xaxis_title='Dataset',
        yaxis_title='Values',
        showlegend=False,
        margin=dict(t=80, b=40)
    )

    fig.show()

def prepare_data(data):
    data['date'] = data['datetime'].dt.date
    data['day'] = data['datetime'].dt.day
    data['year_month'] = data['datetime'].dt.to_period('M')
    return data

def daily_average_plot_split_month(data):
    data = prepare_data(data)
    daily_averages = data.groupby(['year_month', 'date', 'day']).mean().reset_index()

    fig = go.Figure()

    y_min = min(daily_averages['data_0'].min(), daily_averages['data_1'].min()) - 0.5
    y_max = max(daily_averages['data_0'].max(), daily_averages['data_1'].max()) + 0.5

    color_data_0 = 'blue'
    color_data_1 = 'orange'

    for month in daily_averages['year_month'].dt.strftime('%Y-%m').unique():
        monthly_data = daily_averages[daily_averages['year_month'].dt.strftime('%Y-%m') == month]
        fig.add_trace(go.Scatter(
            x=monthly_data['date'], 
            y=monthly_data['data_0'], 
            mode='lines+markers', 
            name=f'Data 0 - {month}', 
            line=dict(dash='solid', color=color_data_0)
        ))
        fig.add_trace(go.Scatter(
            x=monthly_data['date'], 
            y=monthly_data['data_1'], 
            mode='lines+markers', 
            name=f'Data 1 - {month}', 
            line=dict(dash='dash', color=color_data_1)
        ))

    fig.update_layout(
        title_text='Comparison of Monthly Average Data 0 and Data 1 Over Time',
        title_x=0.5,
        height=500,
        width=1000,
        template='plotly_white',
        title_font=dict(size=20, family='Arial, sans-serif')
    )
    fig.update_yaxes(range=[y_min, y_max], title_text='Average Value', title_font=dict(size=14, family='Arial, sans-serif'))
    fig.update_xaxes(title_text='Date', tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'))

    fig.show()

def decomposition_plot(decomposition):
    observed = decomposition.observed
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    resid = decomposition.resid

    # Group by month
    observed_monthly = observed.groupby(observed.index.month)
    trend_monthly = trend.groupby(trend.index.month)
    seasonal_monthly = seasonal.groupby(seasonal.index.month)
    resid_monthly = resid.groupby(resid.index.month)

    # Find the global min and max for each component
    observed_min, observed_max = observed.min(), observed.max()
    trend_min, trend_max = trend.min(), trend.max()
    seasonal_min, seasonal_max = seasonal.min() - 1, seasonal.max() + 1
    resid_min, resid_max = resid.min(), resid.max()

    # Plotting
    for month in observed_monthly.groups.keys():
        plt.figure(figsize=(15, 12))
        
        # Observed component
        plt.subplot(4, 1, 1)
        plt.plot(observed_monthly.get_group(month))
        plt.ylim(observed_min, observed_max)
        plt.title(f'Observed Component for Month {month}')
        
        # Trend component
        plt.subplot(4, 1, 2)
        plt.plot(trend_monthly.get_group(month))
        plt.ylim(trend_min, trend_max)
        plt.title(f'Trend Component for Month {month}')
        
        # Seasonal component
        plt.subplot(4, 1, 3)
        plt.plot(seasonal_monthly.get_group(month))
        plt.ylim(seasonal_min, seasonal_max)
        plt.title(f'Seasonal Component for Month {month}')
        
        # Residual component
        plt.subplot(4, 1, 4)
        plt.plot(resid_monthly.get_group(month))
        plt.ylim(resid_min, resid_max)
        plt.title(f'Residual Component for Month {month}')
        
        plt.tight_layout()
        plt.show()

def day15_split_time_series_plot(data):
    data['value'] = data['datetime'].apply(lambda x: 1 if x.day <= 15 else 2)
    data['year_month'] = data['datetime'].dt.to_period('M')

    months = data['year_month'].unique()
    days_15 = data['value'].unique()

    fig = make_subplots(
        rows=len(months) * len(days_15), cols=2, 
        subplot_titles=[f'Data 0 - {month} (Day {"1-15" if day == 1 else "16-31"})' for month in months for day in days_15] + 
                       [f'Data 1 - {month} (Day {"1-15" if day == 1 else "16-31"})' for month in months for day in days_15],
        vertical_spacing=0.02, horizontal_spacing=0.02
    )

    plot_index = 0
    for month in months:
        for day in days_15:
            daily_data = data[(data['value'] == day) & (data['year_month'] == month)]
            y_min = min(daily_data['data_0'].min(), daily_data['data_1'].min()) - 0.5
            y_max = max(daily_data['data_0'].max(), daily_data['data_1'].max()) + 0.5

            fig.add_trace(
                go.Scatter(x=daily_data['datetime'], y=daily_data['data_0'], mode='lines+markers', name='Data 0',
                           marker=dict(symbol='circle', color='blue', size=8), line=dict(dash='solid', width=2)),
                row=plot_index + 1, col=1
            )
            fig.add_trace(
                go.Scatter(x=daily_data['datetime'], y=daily_data['data_1'], mode='lines+markers', name='Data 1',
                           marker=dict(symbol='square', color='orange', size=8), line=dict(dash='dash', width=2)),
                row=plot_index + 1, col=2
            )

            fig.update_yaxes(range=[y_min, y_max], title_text='Value', title_font=dict(size=14, family='Arial, sans-serif'), row=plot_index + 1, col=1)
            fig.update_yaxes(range=[y_min, y_max], title_text='Value', title_font=dict(size=14, family='Arial, sans-serif'), row=plot_index + 1, col=2)
            fig.update_xaxes(title_text='Datetime', tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'), row=plot_index + 1, col=1)
            fig.update_xaxes(title_text='Datetime', tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'), row=plot_index + 1, col=2)

            plot_index += 1

    fig.update_layout(
        title_text='Comparison of Data 0 and Data 1 Over Time by Month and Day Category',
        title_x=0.5,
        title_font=dict(size=20, family='Arial, sans-serif'),
        height=500 * len(months) * len(days_15),
        width=1200,
        template='plotly_white',
        showlegend=False
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, family='Arial, sans-serif')

    fig.show()

def monthly_average_plot(data):
    monthly_averages = data[["data_0", "data_1", "year_month"]].groupby('year_month').mean().reset_index()
    monthly_averages['year_month'] = pd.PeriodIndex(monthly_averages['year_month'], freq='M').to_timestamp()

    y_min = min(monthly_averages['data_0'].min(), monthly_averages['data_1'].min()) - 0.1
    y_max = max(monthly_averages['data_0'].max(), monthly_averages['data_1'].max()) + 0.1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=monthly_averages['year_month'], y=monthly_averages['data_0'], mode='lines+markers', name='Data 0', marker=dict(symbol='circle', color='blue'), line=dict(dash='solid', color='blue'))
    )
    fig.add_trace(
        go.Scatter(x=monthly_averages['year_month'], y=monthly_averages['data_1'], mode='lines+markers', name='Data 1', marker=dict(symbol='square', color='orange'), line=dict(dash='dash', color='orange'))
    )

    fig.update_layout(
        title_text='Comparison of Data 0 and Data 1 Over Time',
        title_x=0.5,
        title_font=dict(size=20, family='Arial, sans-serif'),
        xaxis_title='Datetime',
        yaxis_title='Value',
        height=500,
        width=900,
        template='plotly_white'
    )

    fig.update_xaxes(tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'))
    fig.update_yaxes(range=[y_min, y_max], title_font=dict(size=14, family='Arial, sans-serif'))

    fig.show()

def monthly_split_time_series_plot(data):
    data = prepare_data(data)
    months = data['year_month'].unique()
    # months = data.index.to_period('M').unique()
    plots = []

    for month in months:
        monthly_data = data[data['year_month'] == month]

        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=(f'data_0 for {month}', f'data_1 for {month}'),
                            vertical_spacing=0.03, horizontal_spacing=0.02)

        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['data_0'], name='data_0', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['data_1'], name='data_1', mode='lines'), row=2, col=1)

        # # Anomalies for data_0
        # anomaly_indices_0 = monthly_data[monthly_data['is_anomaly']].index
        # fig.add_trace(go.Scatter(x=anomaly_indices_0, y=monthly_data.loc[anomaly_indices_0, 'data_0'], 
        #                         mode='markers', name='Anomaly data_0', 
        #                         marker=dict(color='red', size=10)), row=1, col=1)
        
        # # Anomalies for data_1
        # anomaly_indices_1 = monthly_data[monthly_data['is_anomaly']].index
        # fig.add_trace(go.Scatter(x=anomaly_indices_1, y=monthly_data.loc[anomaly_indices_1, 'data_1'], 
        #                         mode='markers', name='Anomaly data_1', 
        #                         marker=dict(color='blue', size=10)), row=2, col=1)

        fig.update_layout(title_text=f"Data for {month}")
        plots.append(fig)

    # Display the plots
    for plot in plots:
        plot.show()

    # data = prepare_data(data)
    # months = data['year_month'].unique()

    # fig = make_subplots(
    #     rows=len(months), cols=2, 
    #     subplot_titles=[f'{m} - Data 0' for m in months] + [f'{m} - Data 1' for m in months],
    #     # vertical_spacing=0.05, 
    #     horizontal_spacing=0.02
    # )

    # for i, month in enumerate(months):
    #     monthly_data = data[data['year_month'] == month]
    #     y_min = min(monthly_data['data_0'].min(), monthly_data['data_1'].min()) - 0.5
    #     y_max = max(monthly_data['data_0'].max(), monthly_data['data_1'].max()) + 0.5

    #     fig.add_trace(
    #         go.Scatter(x=monthly_data['datetime'], y=monthly_data['data_0'], mode='lines+markers', name='Data 0',
    #                    marker=dict(symbol='circle', color='blue')),
    #         row=i+1, col=1
    #     )
    #     fig.add_trace(
    #         go.Scatter(x=monthly_data['datetime'], y=monthly_data['data_1'], mode='lines+markers', name='Data 1',
    #                    marker=dict(symbol='square', color='orange'), line=dict(dash='dash')),
    #         row=i+1, col=2
    #     )

    #     fig.update_yaxes(title_text="Value", range=[y_min, y_max], title_font=dict(size=14, family='Arial, sans-serif'), row=i+1, col=1)
    #     fig.update_yaxes(title_text="Value", range=[y_min, y_max], title_font=dict(size=14, family='Arial, sans-serif'), row=i+1, col=2)
    #     fig.update_xaxes(title_text='Datetime', tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'), row=i+1, col=1)
    #     fig.update_xaxes(title_text='Datetime', tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'), row=i+1, col=2)

    # fig.update_layout(
    #     # height=300 * len(months),
    #     # width=1200,
    #     title_text="Comparison of Data 0 and Data 1 Over Time",
    #     title_x=0.5,
    #     title_font=dict(size=20, family='Arial, sans-serif'),
    #     showlegend=False,
    #     template='plotly_white'
    # )

    # fig.show()

def data_distribution_plot(data):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram for data_0
    axs[0].hist(data['data_0'], bins=100, color='skyblue', edgecolor='black')
    axs[0].set_title('Histogram of data_0')
    axs[0].set_xlabel('data_0')
    axs[0].set_ylabel('Frequency')

    # Histogram for data_1
    axs[1].hist(data['data_1'], bins=100, color='salmon', edgecolor='black')
    axs[1].set_title('Histogram of data_1')
    axs[1].set_xlabel('data_1')
    axs[1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # fig = make_subplots(
    #     rows=1, cols=2, 
    #     subplot_titles=('Distribution of Data 0', 'Distribution of Data 1'),
    #     vertical_spacing=0.02, horizontal_spacing=0.02
    # )

    # fig.add_trace(go.Histogram(
    #     x=data['data_0'],
    #     nbinsx=100,
    #     name='Data 0',
    #     marker_color='blue',
    #     opacity=0.7
    # ), row=1, col=1)
    # fig.add_trace(go.Histogram(
    #     x=data['data_1'],
    #     nbinsx=100,
    #     name='Data 1',
    #     marker_color='orange',
    #     opacity=0.7
    # ), row=1, col=2)

    # fig.update_layout(
    #     title_text='Comparison of Data 0 and Data 1 Distribution',
    #     title_x=0.5,
    #     title_font=dict(size=20, family='Arial, sans-serif'),
    #     template='plotly_white',
    #     font=dict(size=14),
    #     legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    #     margin=dict(t=80, b=40)
    # )

    # fig.update_xaxes(title_text='Value', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=1)
    # fig.update_xaxes(title_text='Value', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=2)
    # fig.update_yaxes(title_text='Frequency', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=1)
    # fig.update_yaxes(title_text='Frequency', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=2)

    # fig.show()

def raw_data_time_series_plot(data):
    y_min = min(data['data_0'].min(), data['data_1'].min()) - 0.5
    y_max = max(data['data_0'].max(), data['data_1'].max()) + 0.5

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Data 0 Over Time", "Data 1 Over Time"),
        vertical_spacing=0.02, horizontal_spacing=0.02
    )

    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['data_0'], mode='lines+markers', name='Data 0',
                   marker=dict(symbol='circle', color='blue', size=8), line=dict(dash='solid', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['data_1'], mode='lines+markers', name='Data 1',
                   marker=dict(symbol='square', color='orange', size=8), line=dict(dash='dash', width=2)),
        row=1, col=2
    )

    fig.update_layout(
        title_text='Comparison of Data 0 and Data 1 Over Time',
        title_x=0.5,
        title_font=dict(size=20, family='Arial, sans-serif'),
        height=500,
        width=1000,
        template='plotly_white'
    )

    fig.update_yaxes(range=[y_min, y_max], title_text='Value', title_font=dict(size=14, family='Arial, sans-serif'))
    fig.update_xaxes(title_text='Datetime', tickangle=45, title_font=dict(size=14, family='Arial, sans-serif'))

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, family='Arial, sans-serif')

    fig.show()

def monthly_raw_time_series_plot(data):
    data['time'] = data['datetime'].dt.strftime('%H:%M')
    data['month'] = data['datetime'].dt.to_period('M')
    data['day'] = data['datetime'].dt.day
    months = data['month'].unique()

    colors = go.Figure().layout.template.layout.colorway
    hourly_ticks = [f'{hour:02}:00' for hour in range(25)]

    for month in months:
        fig = make_subplots(
            rows=1, cols=2, shared_xaxes=True,
            subplot_titles=[f"{month} - Data 0", f"{month} - Data 1"],
            vertical_spacing=0.02, horizontal_spacing=0.02
        )

        month_data = data[data['month'] == month]
        y_min = min(month_data['data_0'].min(), month_data['data_1'].min()) - 5
        y_max = max(month_data['data_0'].max(), month_data['data_1'].max()) + 2
        days = month_data['day'].unique()

        for day in days:
            day_data = month_data[month_data['day'] == day]
            fig.add_trace(go.Scatter(x=day_data['time'], y=day_data['data_0'], mode='lines+markers', name=f'Day {day}', line=dict(color=colors[day % len(colors)])), row=1, col=1)
            fig.add_trace(go.Scatter(x=day_data['time'], y=day_data['data_1'], mode='lines+markers', name=f'Day {day}', line=dict(color=colors[day % len(colors)])), row=1, col=2)

        fig.update_layout(
            height=300,
            width=1000,
            title_text=f"Comparison of Data 0 and Data 1 Over Time for {month}",
            title_x=0.5,
            title_font=dict(size=24, family='Arial, sans-serif'),
            template='plotly_white',
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        fig.update_xaxes(
            tickmode='array', 
            tickvals=hourly_ticks, 
            ticktext=hourly_ticks, 
            title_text='Time of Day',
            title_font=dict(size=14, family='Arial, sans-serif'),
            tickfont=dict(size=12, family='Arial, sans-serif')
        )
        fig.update_yaxes(
            range=[y_min, y_max],
            title_text='Data 0 Value',
            title_font=dict(size=14, family='Arial, sans-serif'),
            tickfont=dict(size=12, family='Arial, sans-serif'),
            row=1, col=1
        )
        fig.update_yaxes(
            range=[y_min, y_max],
            title_text='Data 1 Value',
            title_font=dict(size=14, family='Arial, sans-serif'),
            tickfont=dict(size=12, family='Arial, sans-serif'),
            row=1, col=2
        )

        fig.show()
# def monthly_raw_time_series_plot(data):
#     data['time'] = data['datetime'].dt.strftime('%H:%M')
#     data['month'] = data['datetime'].dt.to_period('M')
#     data['day'] = data['datetime'].dt.day
#     months = data['month'].unique()

#     subplot_titles = sorted([f"{month} - Data 0" for month in months] + [f"{month} - Data 1" for month in months])
#     # subplot_titles.sort(key=lambda x: x.split('- Data ')[-1])
#     # subplot_titles.sort(key=lambda x: x.split(' - ')[0])

#     fig = make_subplots(
#         rows=len(months), cols=2, shared_xaxes=True, 
#         subplot_titles=subplot_titles,
#         vertical_spacing=0.02, horizontal_spacing=0.02
#     )

#     colors = go.Figure().layout.template.layout.colorway
#     hourly_ticks = [f'{hour:02}:00' for hour in range(25)]

#     for i, month in enumerate(months):
#         month_data = data[data['month'] == month]
#         y_min = min(month_data['data_0'].min(), month_data['data_1'].min()) - 5
#         y_max = max(month_data['data_0'].max(), month_data['data_1'].max()) + 2
#         days = month_data['day'].unique()
        
#         for day in days:
#             day_data = month_data[month_data['day'] == day]
#             fig.add_trace(go.Scatter(x=day_data['time'], y=day_data['data_0'], mode='lines+markers', name=f'Day {day}', line=dict(color=colors[day % len(colors)])), row=i+1, col=1)
#             fig.add_trace(go.Scatter(x=day_data['time'], y=day_data['data_1'], mode='lines+markers', name=f'Day {day}', line=dict(color=colors[day % len(colors)])), row=i+1, col=2)

#     fig.update_layout(
#         height=300 * len(months),
#         width=1400,
#         title_text="Comparison of Data 0 and Data 1 Over Time by Month and Day",
#         title_x=0.5,
#         title_font=dict(size=24, family='Arial, sans-serif'),
#         template='plotly_white',
#         showlegend=False,
#         margin=dict(l=40, r=40, t=60, b=40)
#     )

#     for i in range(len(months)):
#         if i == len(months) - 1:
#             fig.update_xaxes(
#                 tickmode='array', 
#                 tickvals=hourly_ticks, 
#                 ticktext=hourly_ticks, 
#                 row=i+1, col=1,
#                 title_text='Time of Day',
#                 title_font=dict(size=14, family='Arial, sans-serif'),
#                 tickfont=dict(size=12, family='Arial, sans-serif')
#             )
#             fig.update_xaxes(
#                 tickmode='array', 
#                 tickvals=hourly_ticks, 
#                 ticktext=hourly_ticks, 
#                 row=i+1, col=2,
#                 title_text='Time of Day',
#                 title_font=dict(size=14, family='Arial, sans-serif'),
#                 tickfont=dict(size=12, family='Arial, sans-serif')
#             )
#         else:
#             fig.update_xaxes(
#                 tickmode='array', 
#                 tickvals=hourly_ticks, 
#                 ticktext=hourly_ticks, 
#                 row=i+1, col=1,
#                 showticklabels=False
#             )
#             fig.update_xaxes(
#                 tickmode='array', 
#                 tickvals=hourly_ticks, 
#                 ticktext=hourly_ticks, 
#                 row=i+1, col=2,
#                 showticklabels=False
#             )
#         fig.update_yaxes(
#             range=[y_min, y_max],
#             title_text='Data 0 Value',
#             title_font=dict(size=14, family='Arial, sans-serif'),
#             tickfont=dict(size=12, family='Arial, sans-serif'),
#             row=i+1, col=1
#         )
#         fig.update_yaxes(
#             range=[y_min, y_max],
#             title_text='Data 1 Value',
#             title_font=dict(size=14, family='Arial, sans-serif'),
#             tickfont=dict(size=12, family='Arial, sans-serif'),
#             row=i+1, col=2
#         )

#     fig.show()

def daily_average_plot(data):
    data = prepare_data(data)
    daily_averages = data.groupby(['year_month', 'date', 'day']).mean().reset_index()

    months = daily_averages['year_month'].unique()

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=('Data 0 - Daily Averages Over Time', 'Data 1 - Daily Averages Over Time')
    )

    for month in months:
        monthly_data = daily_averages[daily_averages['year_month'] == month]
        fig.add_trace(go.Scatter(x=monthly_data['day'], y=monthly_data['data_0'], mode='lines+markers', name=f'Data 0 - {month}', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_data['day'], y=monthly_data['data_1'], mode='lines+markers', name=f'Data 1 - {month}', line=dict(dash='dash', color='orange')), row=1, col=2)

    fig.update_xaxes(title_text='Day', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=1)
    fig.update_yaxes(title_text='Average Value', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=1)
    fig.update_xaxes(title_text='Day', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=2)
    fig.update_yaxes(title_text='Average Value', title_font=dict(size=14, family='Arial, sans-serif'), row=1, col=2)

    fig.update_layout(
        title_text='Comparison of Daily Average Data 0 and Data 1 Over Time',
        title_x=0.5,
        title_y=0.9,
        title_font=dict(size=20, family='Arial, sans-serif'),
        showlegend=True,
        legend_title_text='Months',
        height=600,
        template='plotly_white'
    )

    fig.show()


def check_if_data_is_normal_dist(data):

    def test_normality(data):
        k2, p_value = stats.normaltest(data)
        return p_value

    # Aggregating data by time
    grouped = data.groupby('time').agg(list)

    # Performing normality test for each time instance
    # normality_results = grouped.applymap(test_normality)
    normality_results = grouped.map(lambda x: test_normality(x))


    # Visualizing the distribution of p-values
    # Plotting the distribution of p-values for each sensor
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))

    # Sensor 0
    axs[0].hist(normality_results['data_0'], bins=30, alpha=0.7, label='Sensor 0', color='blue')
    axs[0].axvline(x=0.05, color='r', linestyle='--', label='p = 0.05')
    axs[0].set_xlabel('p-value')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Sensor 0 Data at Each Time Instance')
    axs[0].legend()

    # Sensor 1
    axs[1].hist(normality_results['data_1'], bins=30, alpha=0.7, label='Sensor 1', color='orange')
    axs[1].axvline(x=0.05, color='r', linestyle='--', label='p = 0.05')
    axs[1].set_xlabel('p-value')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Sensor 1 Data at Each Time Instance')
    axs[1].legend()

    fig.suptitle('Distribution of p-values for Normality Test')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()