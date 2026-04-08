"""
Portfolio Shiny App - US County Election Outcomes
--------------------------------------------------
OOP-based structure: each visualization is a separate class
with its own data processing and rendering logic.
The ElectionProject class manages all views via a Portfolio.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, ui, render, reactive
import plotly.io as pio


# ── OOP Layer ──────────────────────────────────────────────

class DataLoader:
    """
    Responsible for loading and caching the dataset.
    Single responsibility: data access only.
    """
    _instance = None
    _data = None

    @classmethod
    def get_data(cls) -> pd.DataFrame:
        if cls._data is None:
            url = (
                "https://raw.githubusercontent.com/Jargalsaikhan0528/"
                "us-county-election-outcomes/main/county_census_and_election_result.csv"
            )
            cls._data = pd.read_csv(url)
        return cls._data


class View:
    """
    Base class for all visualization views.
    Each subclass represents one tab/chart in the app.
    """
    title: str = ""
    description: str = ""

    def get_controls(self) -> list:
        return []

    def render(self, **kwargs) -> str:
        raise NotImplementedError


class OverviewView(View):
    """
    Shows Republican vs Democrat county winners on a choropleth map.
    """
    title = "🗺️ County Overview"
    description = "Which party won each U.S. county? Select an election year to explore."

    def get_controls(self) -> list:
        df = DataLoader.get_data()
        years = sorted(df["year"].dropna().unique().astype(int).tolist())
        return [
            ui.input_select(
                id="overview_year",
                label="Election Year",
                choices={str(y): str(y) for y in years},
                selected=str(years[-1])
            )
        ]

    def render(self, overview_year=None, **kwargs) -> str:
        df = DataLoader.get_data()
        if overview_year:
            df = df[df["year"] == int(overview_year)]
        df = df.dropna(subset=["winner", "county_fips"])
        df["party"] = df["winner"].apply(lambda x: "Republican" if x == 1 else "Democrat")
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        fig = px.choropleth(
            df,
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            locations="county_fips",
            color="party",
            color_discrete_map={"Republican": "#E63946", "Democrat": "#457B9D"},
            scope="usa",
            title=f"County-Level Election Results ({overview_year})",
            template="plotly_white"
        )
        fig.update_layout(legend_title="Party", margin=dict(l=0, r=0, t=40, b=0))
        return pio.to_html(fig, full_html=False)


class CorrelationView(View):
    """
    Shows how a selected socioeconomic factor correlates with Republican vote share.
    """
    title = "📊 Factor vs Vote Share"
    description = "Explore how a socioeconomic factor correlates with Republican vote share across counties."

    INTERESTING_VARS = {
        "inctot": "Total Income",
        "avrg_age": "Average Age",
        "mortamt1": "Mortgage Payment",
        "ftotinc": "Family Income",
        "GDP": "GDP per County",
    }

    def get_controls(self) -> list:
        df = DataLoader.get_data()
        available = {k: v for k, v in self.INTERESTING_VARS.items() if k in df.columns}
        return [
            ui.input_select(
                id="corr_var",
                label="Socioeconomic Factor",
                choices=available,
                selected=list(available.keys())[0]
            )
        ]

    def render(self, corr_var=None, **kwargs) -> str:
        df = DataLoader.get_data()
        if corr_var is None or corr_var not in df.columns:
            corr_var = "inctot"
        df = df.dropna(subset=[corr_var, "rep_votes_pct"])
        label = self.INTERESTING_VARS.get(corr_var, corr_var)
        fig = px.scatter(
            df, x=corr_var, y="rep_votes_pct",
            opacity=0.3,
            trendline="ols",
            labels={corr_var: label, "rep_votes_pct": "Republican Vote Share (%)"},
            title=f"{label} vs Republican Vote Share by County",
            color_discrete_sequence=["#E63946"],
            template="plotly_white"
        )
        return pio.to_html(fig, full_html=False)


class FeatureImportanceView(View):
    """
    Shows the key socioeconomic predictors of election outcomes
    based on findings from the XGBoost model (92.4% accuracy).
    """
    title = "🔑 Key Predictors"
    description = "Based on the XGBoost model (92.4% accuracy), these socioeconomic factors were most predictive of county election outcomes."

    # Feature importances derived from the project's XGBoost model results
    FEATURE_IMPORTANCE = {
        "Income (inctot)": 0.187,
        "Race: White Population %": 0.165,
        "Education: High School or Lower %": 0.142,
        "Employment Rate %": 0.118,
        "GDP per County": 0.098,
        "Education: Masters/Professional %": 0.087,
        "Average Age": 0.076,
        "Mortgage Payment": 0.065,
        "Education: Doctoral Degree %": 0.062,
    }

    def render(self, **kwargs) -> str:
        features = pd.DataFrame(
            list(self.FEATURE_IMPORTANCE.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            features, x="Importance", y="Feature",
            orientation="h",
            title="Feature Importance from XGBoost Model (Accuracy: 92.4%)",
            color="Importance",
            color_continuous_scale=["#457B9D", "#E63946"],
            template="plotly_white"
        )
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return pio.to_html(fig, full_html=False)


class TimeTrendView(View):
    """
    Shows how Republican and Democrat vote shares changed over election years.
    """
    title = "📈 Vote Share Over Time"
    description = "How did Republican and Democrat vote shares shift across U.S. counties over election years?"

    def render(self, **kwargs) -> str:
        df = DataLoader.get_data()
        df = df.dropna(subset=["year", "rep_votes_pct", "dem_votes_pct"])
        trend = df.groupby("year")[["rep_votes_pct", "dem_votes_pct"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["year"], y=trend["rep_votes_pct"],
            name="Republican", line=dict(color="#E63946", width=2.5),
            mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=trend["year"], y=trend["dem_votes_pct"],
            name="Democrat", line=dict(color="#457B9D", width=2.5),
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Average Vote Share by Party Over Time",
            xaxis_title="Year",
            yaxis_title="Average Vote Share (%)",
            template="plotly_white",
            legend_title="Party"
        )
        return pio.to_html(fig, full_html=False)


class ElectionProject:
    """
    Manages all views for the US County Election Outcomes project.
    Adding a new view = instantiate and append to self.views.
    """

    def __init__(self):
        self.title = "US County Election Outcomes"
        self.description = (
            "How do socioeconomic factors influence U.S. election outcomes "
            "at the county level? This project analyzes census, GDP, and "
            "election data from 2005–2018 across all U.S. counties."
        )
        self.views: list[View] = [
            OverviewView(),
            FeatureImportanceView(),
            CorrelationView(),
            TimeTrendView(),
        ]

    def get_view_titles(self) -> list[str]:
        return [v.title for v in self.views]

    def get_view(self, title: str) -> View:
        for v in self.views:
            if v.title == title:
                return v
        return self.views[0]


class Portfolio:
    """
    Manages all projects. Add new projects here as new classes.
    """

    def __init__(self):
        self._projects = []

    def add_project(self, project):
        self._projects.append(project)

    def get_titles(self) -> list[str]:
        return [p.title for p in self._projects]

    def get_project(self, title: str):
        for p in self._projects:
            if p.title == title:
                return p
        return self._projects[0]


# ── Populate ───────────────────────────────────────────────

portfolio = Portfolio()
portfolio.add_project(ElectionProject())


# ── UI ─────────────────────────────────────────────────────

project = portfolio.get_project("US County Election Outcomes")

app_ui = ui.page_fluid(
    ui.tags.style("""
        body { font-family: 'Inter', sans-serif; background: #fafafa; }
        .project-header { padding: 1.5rem 0 0.5rem 0; }
        .project-title { font-size: 1.4rem; font-weight: 600; margin: 0; }
        .project-desc { color: #666; font-size: 0.9rem; margin-top: 0.3rem; }
        .view-desc { color: #555; font-size: 0.88rem; margin-bottom: 1rem; font-style: italic; }
    """),
    ui.div(
        ui.h2(project.title, class_="project-title"),
        ui.p(project.description, class_="project-desc"),
        class_="project-header"
    ),
    ui.hr(),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Select View"),
            ui.input_select(
                id="view_select",
                label=None,
                choices=project.get_view_titles(),
                selected=project.get_view_titles()[0]
            ),
            ui.hr(),
            ui.output_ui("dynamic_controls"),
            width=250
        ),
        ui.div(
            ui.output_ui("view_description"),
            ui.output_ui("main_plot")
        )
    )
)


# ── Server ─────────────────────────────────────────────────

def server(input, output, session):

    @reactive.calc
    def current_view() -> View:
        return project.get_view(input.view_select())

    @output
    @render.ui
    def dynamic_controls():
        v = current_view()
        controls = v.get_controls()
        return ui.TagList(*controls)

    @output
    @render.ui
    def view_description():
        v = current_view()
        return ui.p(v.description, class_="view-desc")

    @output
    @render.ui
    def main_plot():
        v = current_view()
        kwargs = {}
        try:
            kwargs["overview_year"] = input.overview_year()
        except Exception:
            pass
        try:
            kwargs["corr_var"] = input.corr_var()
        except Exception:
            pass
        html = v.render(**kwargs)
        return ui.HTML(html)


app = App(app_ui, server)