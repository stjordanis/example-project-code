import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from bokeh.models import LogColorMapper
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from bokeh.sampledata.unemployment import data as unemployment
from bokeh.sampledata.us_counties import data as counties
from neptunecontrib.api import log_chart
from vega_datasets import data

pltt = palette
cnts = counties


def log_interactive_visualisations():
    _log_matplotlib_figure()
    _log_plotly_figure()
    _log_bokeh_figure()
    _log_altair_figure()


def _log_matplotlib_figure():
    def _scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # some random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # use the previously defined function
    _scatter_hist(x, y, ax, ax_histx, ax_histy)

    # Log matplotlib figure as interactive chart in the experiment' artifacts tab.
    log_chart(name='matplotlib_figure', chart=fig)
    plt.close('all')


def _log_plotly_figure():
    df = px.data.tips()
    plotly_fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug",
                              hover_data=df.columns)

    # Log plotly figure as interactive chart in the experiment' artifacts tab.
    log_chart(name='plotly_figure', chart=plotly_fig)


def _log_bokeh_figure():
    palette2 = tuple(reversed(pltt))

    counties = {
        code: county for code, county in cnts.items() if county["state"] == "tx"
    }

    county_xs = [county["lons"] for county in counties.values()]
    county_ys = [county["lats"] for county in counties.values()]

    county_names = [county['name'] for county in counties.values()]
    county_rates = [unemployment[county_id] for county_id in counties]
    color_mapper = LogColorMapper(palette=palette2)

    chart_data = dict(
        x=county_xs,
        y=county_ys,
        name=county_names,
        rate=county_rates,
    )

    TOOLS = "pan,wheel_zoom,reset,hover,save"

    p = figure(
        title="Texas Unemployment, 2009", tools=TOOLS,
        x_axis_location=None, y_axis_location=None,
        tooltips=[
            ("Name", "@name"), ("Unemployment rate", "@rate%"), ("(Long, Lat)", "($x, $y)")
        ])
    p.grid.grid_line_color = None
    p.hover.point_policy = "follow_mouse"

    p.patches('x', 'y', source=chart_data,
              fill_color={'field': 'rate', 'transform': color_mapper},
              fill_alpha=0.7, line_color="white", line_width=0.5)

    # Log bokeh figure as interactive chart in the experiment' artifacts tab.
    log_chart(name='bokeh_figure', chart=p)


def _log_altair_figure():
    source = data.cars()

    brush = alt.selection(type='interval')

    points = alt.Chart(source).mark_point().encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color=alt.condition(brush, 'Origin:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )

    bars = alt.Chart(source).mark_bar().encode(
        y='Origin:N',
        color='Origin:N',
        x='count(Origin):Q'
    ).transform_filter(
        brush
    )

    chart = points & bars

    # Log altair figure as interactive chart in the experiment' artifacts tab.
    log_chart(name='altair_chart', chart=chart)
