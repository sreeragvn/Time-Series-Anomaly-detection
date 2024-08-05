import plotly.graph_objects as go
import plotly.subplots as sp

def multivariate_anomaly_plot(data, diff=False):
    months = data.index.to_period('M').unique()
    plots = []

    for month in months:
        monthly_data = data[data.index.to_period('M') == month]

        if diff == False:
            fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                subplot_titles=(f'data_0 for {month}', f'data_1 for {month}'),
                                vertical_spacing=0.03, horizontal_spacing=0.02)
        else:
            fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                subplot_titles=(f'data_0 for {month}', f'data_1 for {month}'),
                                vertical_spacing=0.03, horizontal_spacing=0.02)

        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['data_0'], name='data_0', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['data_1'], name='data_1', mode='lines'), row=2, col=1)
        if diff == True:
            fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['diff'], name='diff', mode='lines'), row=3, col=1)

        # Anomalies for data_0
        anomaly_indices_0 = monthly_data[monthly_data['is_anomaly']].index
        fig.add_trace(go.Scatter(x=anomaly_indices_0, y=monthly_data.loc[anomaly_indices_0, 'data_0'], 
                                mode='markers', name='Anomaly data_0', 
                                marker=dict(color='red', size=10)), row=1, col=1)
        
        # Anomalies for data_1
        anomaly_indices_1 = monthly_data[monthly_data['is_anomaly']].index
        fig.add_trace(go.Scatter(x=anomaly_indices_1, y=monthly_data.loc[anomaly_indices_1, 'data_1'], 
                                mode='markers', name='Anomaly data_1', 
                                marker=dict(color='blue', size=10)), row=2, col=1)
        
        # # Add vertical lines for anomalies in data_0
        # anomaly_indices_0 = monthly_data[monthly_data['data_0_anomaly'] == 1].index
        # for anomaly_index in anomaly_indices_0:
        #     fig.add_vline(x=anomaly_index, line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dot'), row=1, col=1)

        # # Add vertical lines for anomalies in data_1
        # anomaly_indices_1 = monthly_data[monthly_data['data_1_anomaly'] == 1].index
        # for anomaly_index in anomaly_indices_1:
        #     fig.add_vline(x=anomaly_index, line=dict(color='rgba(0, 0, 255, 0.5)', width=1, dash='dot'), row=2, col=1)


        fig.update_layout(title_text=f"Data for {month}")
        plots.append(fig)

    # Display the plots
    for plot in plots:
        plot.show()


def univariate_anomaly_plot(data, diff=False):
    months = data.index.to_period('M').unique()
    plots = []

    for month in months:
        monthly_data = data[data.index.to_period('M') == month]

        if diff == False:
            fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                subplot_titles=(f'data_0 for {month}', f'data_1 for {month}'),
                                vertical_spacing=0.03, horizontal_spacing=0.02)
        else:
            fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                subplot_titles=(f'data_0 for {month}', f'data_1 for {month}'),
                                vertical_spacing=0.03, horizontal_spacing=0.02)


        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['data_0'], name='data_0', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['data_1'], name='data_1', mode='lines'), row=2, col=1)
        if diff == True:
            fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['diff'], name='diff', mode='lines'), row=3, col=1)

        # Anomalies for data_0
        anomaly_indices_0 = monthly_data[monthly_data['is_anomaly_0']].index
        fig.add_trace(go.Scatter(x=anomaly_indices_0, y=monthly_data.loc[anomaly_indices_0, 'data_0'], 
                                mode='markers', name='Anomaly data_0', 
                                marker=dict(color='red', size=10)), row=1, col=1)
        
        # Anomalies for data_1
        anomaly_indices_1 = monthly_data[monthly_data['is_anomaly_1']].index
        fig.add_trace(go.Scatter(x=anomaly_indices_1, y=monthly_data.loc[anomaly_indices_1, 'data_1'], 
                                mode='markers', name='Anomaly data_1', 
                                marker=dict(color='blue', size=10)), row=2, col=1)
        
        # # Add vertical lines for anomalies in data_0
        # anomaly_indices_0 = monthly_data[monthly_data['data_0_anomaly'] == 1].index
        # for anomaly_index in anomaly_indices_0:
        #     fig.add_vline(x=anomaly_index, line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dot'), row=1, col=1)

        # # Add vertical lines for anomalies in data_1
        # anomaly_indices_1 = monthly_data[monthly_data['data_1_anomaly'] == 1].index
        # for anomaly_index in anomaly_indices_1:
        #     fig.add_vline(x=anomaly_index, line=dict(color='rgba(0, 0, 255, 0.5)', width=1, dash='dot'), row=2, col=1)


        fig.update_layout(title_text=f"Data for {month}")
        plots.append(fig)

    # Display the plots
    for plot in plots:
        plot.show()

def multivariate_anomaly_day_plot(data):
    months = data.index.to_period('M').unique()
    plots = []

    for month in months:
        monthly_data = data[data.index.to_period('M') == month].sort_index()
        days = monthly_data.index.to_period('D').unique()

        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=(f'data_0 for {month}', f'data_1 for {month}'),
                            vertical_spacing=0.03, horizontal_spacing=0.02)
        
        for day in days:
            daily_data = monthly_data[monthly_data.index.to_period('D') == day]
            
            fig.add_trace(go.Scatter(
                x=daily_data.index.time,
                y=daily_data['data_0'],
                name='data_0',
                mode='lines',
                customdata=daily_data.index.strftime('%Y-%m-%d'),
                hovertemplate='%{customdata} %{x}<br>Value: %{y}'),
                row=1, col=1
            )
            
            fig.add_trace(go.Scatter(
                x=daily_data.index.time,
                y=daily_data['data_1'],
                name='data_1',
                mode='lines',
                customdata=daily_data.index.strftime('%Y-%m-%d'),
                hovertemplate='%{customdata} %{x}<br>Value: %{y}'),
                row=2, col=1
            )

            # Anomalies for data_0
            anomaly_indices_0 = daily_data[daily_data['is_anomaly']].index
            fig.add_trace(go.Scatter(
                x=anomaly_indices_0.time,
                y=daily_data.loc[anomaly_indices_0, 'data_0'],
                mode='markers',
                name='Anomaly data_0',
                marker=dict(color='red', size=10),
                customdata=anomaly_indices_0.strftime('%Y-%m-%d'),
                hovertemplate='%{customdata} %{x}<br>Value: %{y}'),
                row=1, col=1
            )
            
            # Anomalies for data_1
            anomaly_indices_1 = daily_data[daily_data['is_anomaly']].index
            fig.add_trace(go.Scatter(
                x=anomaly_indices_1.time,
                y=daily_data.loc[anomaly_indices_1, 'data_1'],
                mode='markers',
                name='Anomaly data_1',
                marker=dict(color='blue', size=10),
                customdata=anomaly_indices_1.strftime('%Y-%m-%d'),
                hovertemplate='%{customdata} %{x}<br>Value: %{y}'),
                row=2, col=1
            )

        fig.update_layout(title_text=f"Data for {month}")
        plots.append(fig)

    # Display the plots
    for plot in plots:
        plot.show()