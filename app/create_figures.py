# Created by timot at 05/03/2021
import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine


# TODO: Scroll down to line 157 and set up a fifth visualization for the data dashboard

def cleandata():
    """
    Import cleaned data for us in visualisations
    :return: dataframe
    """
    engine = create_engine('sqlite:///../data/disaster_db.db')
    df = pd.read_sql_table('disaster_db', engine)

    return df


def return_figures():
    """
    Creates the plotly visualisations
    :return: a list of the visualisations
    """
    graph_one = []
    df = cleandata()

    graph_one.append(
            go.Bar(name='Ones', x=['Related', 'Request', 'Offer',
                                   'Aid related', 'Medical help', 'Medical products',
                                   'Search and rescue', 'Security', 'Military', 'Child alone',
                                   'Water', 'Food', 'Shelter', 'Clothing', 'Money', 'Missing people',
                                   'Refugees', 'Death', 'Other aid', 'Infrastructure related',
                                   'Transport', 'Buildings', 'Electricity', 'Tools', 'Hospitals',
                                   'Shops', 'Aid centers', 'Other infrastructure', 'Weather related',
                                   'Floods', 'Storm', 'Fire', 'Earthquake', 'Cold', 'Other weather',
                                   'Direct report'], y=[df['related'].sum(),
                                                             df['request'].sum(),
                                                             df['offer'].sum(),
                                                             df['aid_related'].sum(),
                                                             df['medical_help'].sum(),
                                                             df['medical_products'].sum(),
                                                             df['search_and_rescue'].sum(),
                                                             df['security'].sum(),
                                                             df['military'].sum(),
                                                             df['child_alone'].sum(),
                                                             df['water'].sum(),
                                                             df['food'].sum(),
                                                             df['shelter'].sum(),
                                                             df['clothing'].sum(),
                                                             df['money'].sum(),
                                                             df['missing_people'].sum(),
                                                             df['refugees'].sum(),
                                                             df['death'].sum(),
                                                             df['other_aid'].sum(),
                                                             df['infrastructure_related'].sum(),
                                                             df['transport'].sum(),
                                                             df['buildings'].sum(),
                                                             df['electricity'].sum(),
                                                             df['tools'].sum(),
                                                             df['hospitals'].sum(),
                                                             df['shops'].sum(),
                                                             df['aid_centers'].sum(),
                                                             df['other_infrastructure'].sum(),
                                                             df['weather_related'].sum(),
                                                             df['floods'].sum(),
                                                             df['storm'].sum(),
                                                             df['fire'].sum(),
                                                             df['earthquake'].sum(),
                                                             df['cold'].sum(),
                                                             df['other_weather'].sum(),
                                                             df['direct_report'].sum()]),
            )

    layout_one = dict(title='Distribution of message categories',
                      xaxis=dict(tickangle=45),
                      yaxis=dict(title='Count'),
                      )

    graph_two = []
    graph_two.append(
        go.Bar(
            x=['Direct', 'News', 'Social'],
            y=df.groupby('genre').count()['message'],
        )
    )

    layout_two = dict(title='Distribution of message genres',
                      xaxis=dict(title='Message Genres', ),
                      yaxis=dict(title='Count'),
                      )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures
