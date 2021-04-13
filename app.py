import dash_core_components as dcc
import dash
import dash_html_components as html
from dash_table import DataTable, FormatTemplate
import pandas as pd
import strategy as strat


strat.run_strategy('IVV', 500, pd.Timestamp("2019-01-01"), 0.02, .03, 1.45, 100000)

df_blotter = pd.read_csv('strategy_files/blotter.csv')
df_ledger = pd.read_csv('strategy_files/ledger.csv')
df_portfolio = pd.read_csv('strategy_files/portfolio.csv')

# Create a Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(
        'Blotter Table',
        style={'display': 'block', 'text-align': 'center'}
    ),
    DataTable(
        id='blotter_table',
        columns=[{"name": i, "id": i} for i in df_blotter.columns],
        data=df_blotter.to_dict('records'),
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    # Line break
    html.Br(),
    # Line break
    html.Br(),
    html.H1(
        'Ledger Table',
        style={'display': 'block', 'text-align': 'center'}
    ),
    DataTable(
        id='ledger_table',
        columns=[{"name": i, "id": i} for i in df_ledger.columns],
        data=df_ledger.to_dict('records'),
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    # Line break
    html.Br(),
    # Line break
    html.Br(),
    html.H1(
        'Portfolio Table',
        style={'display': 'block', 'text-align': 'center'}
    ),
    DataTable(
        id='portfolio_table',
        columns=[{"name": i, "id": i} for i in df_portfolio.columns],
        data=df_portfolio.to_dict('records'),
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
