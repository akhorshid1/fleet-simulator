import numpy as np
import pandas as pd
from datetime import datetime
import copy
import json
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go


class FleetSimulator:
    def __init__(self, 
                 min_soc=0.1, 
                 max_soc=1.0,
                 start_soc=0.1):
        self.seed = 42
        self.buy_sell_thresholds = [
            (60, 80), (60, 90), (60, 100), (60, 110), (60, 120), (60, 130),
            (70, 90), (70, 100), (70, 110), (70, 120), (70, 130),
            (80, 100), (80, 110), (80, 120), (80, 130),
            (90, 110), (90, 120), (90, 130)
        ]
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.start_soc = start_soc

    def simulate_energy_prices(self, 
                               start_date, 
                               num_days,
                               resolution=None):
        np.random.seed(self.seed)
        index = pd.date_range(start=start_date, periods=num_days * 24, freq='1H')
        hours = index.hour
        base_price = (
            50
            + 50 * np.exp(-0.5 * ((hours - 7) / 2.0)**2)  # wider but smaller peak at 7am
            + 240 * np.exp(-0.5 * ((hours - 18) / 1.0)**2)  # narrower but larger peak at 6pm
        )
        noise = np.random.normal(loc=0, scale=5, size=len(index))
        price = np.clip(base_price + noise, 50, 250)

        df_prices = pd.DataFrame({"price_$MWh": price}, index=index)

        if resolution is not None:
            periods = 3600 / pd.to_timedelta(resolution).total_seconds()
            # option to increase frequency but keep the intra hour prices the same
            index = pd.date_range(start=start_date, periods=num_days * 24 * periods, freq=resolution)
            df_prices = pd.merge(df_prices, pd.DataFrame(index=index), how='right', left_index=True, right_index=True)
            df_prices = df_prices.fillna(method='ffill')

        return df_prices

    def calculate_total_revenue(self,
                                df_prices,
                                buy_threshold,
                                sell_threshold,
                                **kwargs):

        capacity_mwh = kwargs.get('capacity_mwh')
        charge_rate_mw = kwargs.get('charge_rate_mw')
        discharge_rate_mw = kwargs.get('discharge_rate_mw')
        efficiency = kwargs.get('efficiency')

        soc = capacity_mwh * self.start_soc
        charge_rate_mw = charge_rate_mw * efficiency
        resolution_hour = pd.to_timedelta(df_prices.index.freq).total_seconds() / 3600
        total_revenue = 0

        for price in df_prices['price_$MWh']:
            action, revenue = 0, 0

            if price <= buy_threshold and soc < capacity_mwh * self.max_soc:
                action = min(charge_rate_mw * resolution_hour, capacity_mwh * self.max_soc - soc)
                revenue = (action * -price) / efficiency
                soc += action

            elif price >= sell_threshold and soc > capacity_mwh * self.min_soc:
                action = -min(discharge_rate_mw * resolution_hour, soc - capacity_mwh * self.min_soc)
                revenue = -action * price
                soc += action

            total_revenue += revenue

        return total_revenue

    def generate_fleet_dispatch_curve(self, 
                                      df_prices, 
                                      assets, 
                                      max_fleet_charge_rate_mw, 
                                      max_fleet_discharge_rate_mw):
        
        resolution_hour = pd.to_timedelta(df_prices.index.freq).total_seconds() / 3600
        time_index = df_prices.index
        n_steps = len(df_prices)

        state = {
            name: {
                'soc': config['capacity_mwh'] * self.start_soc,
                'capacity_mwh': config['capacity_mwh'],
                # charge_rate_mwh is a measure of how much energy we can store in t=resolution_hour if charging at charge_rate_mw
                'charge_rate_mwh': config['charge_rate_mw'] * resolution_hour,
                # discharge_rate_mwh is a measure of how much energy we can discharge in t=resolution_hour if discharging at charge_rate_mwh
                'discharge_rate_mwh': config['discharge_rate_mw'] * resolution_hour,
                'efficiency': config['efficiency'],
                'buy_threshold': config['buy_threshold'],
                'sell_threshold': config['sell_threshold']
            } for name, config in assets
        }

        result_records = []

        # prioritize the batteries with the best efficiency which will
        # a) improve revenue during charging
        # b) utilize batteries with better health more often reducing the number of cycles on older batteries
        state = dict(sorted(state.items(), key=lambda item: item[1]['efficiency'], reverse=True))

        for t in range(n_steps):
            price = df_prices.iloc[t]['price_$MWh']
            timestamp = time_index[t]

            # keep track of the fleet charge available when we are allocating power to each battery
            fleet_charge_available = max_fleet_charge_rate_mw * resolution_hour
            fleet_discharge_available = max_fleet_discharge_rate_mw * resolution_hour

            for i, (name, config) in enumerate(state.items()):

                soc, cap, eff = config['soc'], config['capacity_mwh'], config['efficiency']
                min_soc, max_soc = cap * self.min_soc, cap * self.max_soc
                action, revenue = 0, 0

                if price <= config['buy_threshold'] and soc < max_soc:
                    charge_possible = min(config['charge_rate_mwh'], max_soc - soc)  # check if battery is not full and determine how much energy we can add to it
                    charge_allowed = min(charge_possible, fleet_charge_available)  # check if we have charge allocation from the limit set on the fleet
                    energy_stored = charge_allowed * eff
                    energy_bought = energy_stored / eff


                    if charge_allowed > 0:
                        soc += energy_stored  # add energy into battery
                        revenue = -energy_bought * price  # calculate price paid
                        action = energy_bought  # log action taken if charge_allowed > 0
                        fleet_charge_available -= energy_bought

                elif price >= config['sell_threshold'] and soc > min_soc:
                    discharge_possible = min(config['discharge_rate_mwh'], soc - min_soc)  # check if battery is not empty and determine how much energy we can dispatch
                    discharge_allowed = min(discharge_possible, fleet_discharge_available)  # check if we have discharge allocation from the limit set on the fleet

                    if discharge_allowed > 0:
                        soc -= discharge_allowed  # discharge energy from battery
                        revenue = discharge_allowed * price  # calculate revenue made
                        action = -discharge_allowed
                        fleet_discharge_available -= discharge_allowed

                config['soc'] = soc

                result_records.append({
                    'time': timestamp,
                    'asset': name,
                    'price_$MWh': price,
                    'soc_mwh': soc,
                    'soc_percent': soc / cap * 100,
                    'dispatch_mw': action / resolution_hour,
                    'revenue_$': revenue,
                    'efficiency': eff
                })

        result_df = pd.DataFrame(result_records)
        result_df.set_index('time', inplace=True)
        return result_df

    def optimize_assets(self, assets):
        optimized_assets = copy.deepcopy(assets)
        # optimize assets over a short period of 2 days for better efficiency
        df_prices = self.simulate_energy_prices(start_date=datetime.today().strftime('%Y-%m-%d'), num_days=2)
        # this is a greedy algorithm that goes through different buy and sell thresholds to compute the revenue
        # then select the strategy with maximum theoretical revenue.
        # it does not take into account limits on total fleet charge and discharge
        for i, (name, config) in enumerate(optimized_assets):
            scenario_revenues = []

            for buy, sell in self.buy_sell_thresholds:
                revenue = self.calculate_total_revenue(
                    df_prices,
                    buy_threshold=buy,
                    sell_threshold=sell,
                    **config
                )
                scenario_revenues.append(revenue)

            best = np.argmax(scenario_revenues)
            optimized_assets[i][1]['buy_threshold'] = self.buy_sell_thresholds[best][0]
            optimized_assets[i][1]['sell_threshold'] = self.buy_sell_thresholds[best][1]

        return optimized_assets
           
app = dash.Dash(__name__)

# Load assets
with open('assets.json', 'r') as f:
    assets_dict = json.load(f)
asset_names = list(assets_dict.keys())

app.layout = html.Div([
    html.H2("Battery Fleet Simulator"),

    html.Div([
        html.Div([
            html.Label("Start Date: ", style={'marginBottom': '5px'}),
            dcc.DatePickerSingle(
                id='start-date',
                date=datetime.today().strftime('%Y-%m-%d')
            )
        ]),

        html.Div([
            html.Label("Simulation Days: ", style={'marginBottom': '5px'}),
            dcc.Input(id='num-days', type='number', value=2)
        ]),

        html.Div([
            html.Label("Fleet Max Charge Rate (MW): ", style={'marginBottom': '5px'}),
            dcc.Input(id='max-charge', type='number', value=10)
        ]),

        html.Div([
            html.Label("Fleet Max Discharge Rate (MW): ", style={'marginBottom': '5px'}),
            dcc.Input(id='max-discharge', type='number', value=30)
        ]),

        html.Div([
            html.Label("Select Asset: ", style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='asset-select',
                options=[{'label': name, 'value': name} for name in asset_names],
                value=asset_names[0],
                style={'width': '200px', 'fontSize': '14px'}
            )
        ]),

        html.Div([
            html.Button('Run Simulation', id='simulate-btn', n_clicks=0)
        ], style={'display': 'flex', 'alignItems': 'end'})
    ], style={
        'display': 'grid',
        'gridTemplateColumns': '2fr 2fr',
        'gap': '20px',
        'marginBottom': '30px'
    }),

    html.Div([
        html.Div([
            html.H3("Asset Summary Table", style={'marginBottom': '15px'}),
            html.Div(id='summary-table')
        ], style={
            'border': '1px solid #ccc',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),

        html.Div([
            dcc.Graph(id='simulation-graph')
        ], style={
            'border': '1px solid #ccc',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ], style={
        'display': 'flex',
        'flexDirection': 'column'
    })
])

@app.callback(
    Output('simulation-graph', 'figure'),
    Output('summary-table', 'children'),
    Input('simulate-btn', 'n_clicks'),
    Input('asset-select', 'value'),
    Input('num-days', 'value'),
    Input('start-date', 'date'),
    Input('max-charge', 'value'),
    Input('max-discharge', 'value')
)
def run_simulation(n_clicks, selected_asset, num_days, start_date, max_charge, max_discharge):
    simulator = FleetSimulator()
    df_prices = simulator.simulate_energy_prices(start_date=start_date, num_days=num_days)
    optimized_assets = simulator.optimize_assets(list(assets_dict.items()))

    dispatch_df = simulator.generate_fleet_dispatch_curve(
        df_prices,
        optimized_assets,
        max_fleet_charge_rate_mw=max_charge,
        max_fleet_discharge_rate_mw=max_discharge
    )

    fleet_revenue_df = dispatch_df.groupby('time')['revenue_$'].sum().cumsum().reset_index()
    fleet_dispatch_df = dispatch_df.groupby('time')['dispatch_mw'].sum().reset_index()

    df = dispatch_df[dispatch_df['asset'] == selected_asset].copy()
    df['cumulative_revenue'] = df['revenue_$'].cumsum()

    # Create subplot structure with secondary y for the second subplot (row 2)
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=["Electricity Price", "State of Charge", "Battery Dispatch", "Fleet Dispatch", "Cumulative Revenue"],
        specs=[[{}], [{"secondary_y": True}], [{}], [{}], [{}]]
    )

    # Row 1: Price
    fig.add_trace(go.Scatter(x=df.index, y=df['price_$MWh'], name='Price ($/MWh)'), row=1, col=1)

    # Row 2: SoC MWh (primary y-axis)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["soc_mwh"], name="SoC (MWh)", line=dict(color="blue")),
        row=2, col=1, secondary_y=False
    )

    # Row 2: SoC % (secondary y-axis)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["soc_percent"], name="SoC (%)", line=dict(color="orange", dash="dot")),
        row=2, col=1, secondary_y=True
    )

    # Row 3: Asset Dispatch
    fig.add_trace(go.Scatter(x=df.index, y=df['dispatch_mw'], name='Dispatch (MW)'), row=3, col=1)

    # Row 4: Fleet Dispatch
    fig.add_trace(
        go.Scatter(x=fleet_dispatch_df['time'], y=fleet_dispatch_df['dispatch_mw'], name='Total Dispatch (MW)'), row=4,
        col=1)

    # Row 5: Cumulative Revenue
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_revenue'], name='Cumulative Revenue ($)'), row=5, col=1)
    fig.add_trace(go.Scatter(x=fleet_revenue_df['time'], y=fleet_revenue_df['revenue_$'], name='Fleet Revenue ($)'),
                  row=5, col=1)

    # Add axis titles for each subplot
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="SoC (MWh)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SoC (%)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Battery Dispatch (MW)", row=3, col=1)
    fig.update_yaxes(title_text="Fleet Dispatch (MW)", row=4, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=5, col=1)

    # Ensure all x-axes show ticks and labels
    for i in range(1, 6):
        fig.update_xaxes(
            showline=True, linewidth=1, linecolor='black',
            ticks="outside", tickangle=45,
            showticklabels=True, row=i, col=1
        )
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', ticks="outside", row=i, col=1)

    # Label the secondary y-axis for SoC (%)
    fig.update_yaxes(title_text="SoC (MWh)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="SoC (%)", row=2, col=1, secondary_y=True)

    # Overall layout
    fig.update_layout(
        height=1400,
        title_text=f"Simulation Results for {selected_asset}",
        showlegend=True,
        margin=dict(t=40, b=40, l=40, r=40),
        font=dict(
            family="Lato, sans-serif",  # Use any web-safe font or Google Font
            size=14,
            color="black"
        )
    )
    
    fig.update_xaxes(tickangle=45)

    summary = dispatch_df.groupby('asset').agg({
        'revenue_$': 'sum'
    }).rename(columns={'revenue_$': 'Total Revenue ($)'}).round(2).reset_index()

    for asset_name, config in optimized_assets:
        summary.loc[summary['asset'] == asset_name, 'Capacity (MWh)'] = config['capacity_mwh']
        summary.loc[summary['asset'] == asset_name, 'Charge Rate (MW)'] = config['charge_rate_mw']
        summary.loc[summary['asset'] == asset_name, 'Discharge Rate (MW)'] = config['discharge_rate_mw']
        summary.loc[summary['asset'] == asset_name, 'Efficiency'] = config['efficiency']
        summary.loc[summary['asset'] == asset_name, 'Buy Threshold'] = config['buy_threshold']
        summary.loc[summary['asset'] == asset_name, 'Sell Threshold'] = config['sell_threshold']

    total_row = {
        'asset': 'TOTAL',
        'Total Revenue ($)': summary['Total Revenue ($)'].sum().round(2),
        'Charge Rate (MW)': '',
        'Discharge Rate (MW)': '',
        'Efficiency': '',
        'Buy Threshold': '',
        'Sell Threshold': ''
    }

    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    summary_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in summary.columns],
        data=summary.to_dict('records'),
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f2f2f2'},
        style_table={'overflowX': 'auto'},
    )

    return fig, summary_table

if __name__ == '__main__':
    app.run(debug=True)
