"""
Portfolio Shiny App - US County Election Outcomes
--------------------------------------------------
OOP-based structure: each visualization is a separate class.
All views are displayed at once on a single scrollable page.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, ui, render, reactive
import plotly.io as pio


# ── OOP Layer ──────────────────────────────────────────────

class DataLoader:
    """Loads, merges, and caches the dataset."""
    _data = None

    @classmethod
    def get_data(cls) -> pd.DataFrame:
        if cls._data is None:
            base_url = (
                "https://raw.githubusercontent.com/Jargalsaikhan0528/"
                "us-county-election-outcomes/main/"
            )
            df = pd.read_csv(base_url + "county_census_and_election_result.csv")
            gdp = pd.read_csv(base_url + "GDP%20by%20County.csv")

            # Merge with GDP
            gdp = gdp.rename(columns={"Year": "year", "County FIPS": "county_fips", "GDP (Chained $)": "GDP"})
            df = df.merge(gdp[["year", "county_fips", "GDP"]], on=["year", "county_fips"], how="left")

            # Compute vote shares
            df["total_votes"] = df["democrat"] + df["republican"]
            df["republican_share"] = (df["republican"] / df["total_votes"]) * 100
            df["democrat_share"] = (df["democrat"] / df["total_votes"]) * 100

            cls._data = df
        return cls._data


class View:
    """Base class for all visualization views."""
    title: str = ""
    description: str = ""

    def get_controls(self) -> list:
        return []

    def render(self, **kwargs) -> str:
        raise NotImplementedError


class OverviewView(View):
    """Choropleth map of Republican vs Democrat county winners."""
    title = "County Overview"
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


class FeatureImportanceView(View):
    """Key predictors from the XGBoost model."""
    title = "Key Predictors"
    description = "Based on the XGBoost model (92.4% accuracy), these socioeconomic factors were most predictive of county election outcomes."

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


class CorrelationView(View):
    """Correlation between a socioeconomic factor and Republican vote share."""
    title = "Factor vs Vote Share"
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
        df = df.dropna(subset=[corr_var, "republican_share"])
        label = self.INTERESTING_VARS.get(corr_var, corr_var)
        fig = px.scatter(
            df, x=corr_var, y="republican_share",
            opacity=0.3,
            trendline="ols",
            labels={corr_var: label, "republican_share": "Republican Vote Share (%)"},
            title=f"{label} vs Republican Vote Share by County",
            color_discrete_sequence=["#E63946"],
            template="plotly_white"
        )
        return pio.to_html(fig, full_html=False)


class TimeTrendView(View):
    """Vote share trends over election years."""
    title = "Vote Share Over Time"
    description = "How did Republican and Democrat vote shares shift across U.S. counties over election years?"

    def render(self, **kwargs) -> str:
        df = DataLoader.get_data()
        df = df.dropna(subset=["year", "republican_share", "democrat_share"])
        trend = df.groupby("year")[["republican_share", "democrat_share"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["year"], y=trend["republican_share"],
            name="Republican", line=dict(color="#E63946", width=2.5),
            mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=trend["year"], y=trend["democrat_share"],
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
    To add a new visualization: create a new View subclass and append it.
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


class PlaceholderProject:
    """Template for future projects."""
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.views = []


class Portfolio:
    """Manages all projects. Add new projects here."""

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
election = ElectionProject()
portfolio.add_project(election)
portfolio.add_project(PlaceholderProject("Project 2", "Coming soon."))
portfolio.add_project(PlaceholderProject("Project 3", "Coming soon."))


# ── UI ─────────────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.tags.style("""
        body { font-family: 'Inter', sans-serif; background: #fafafa; }
        .project-header { padding: 1.5rem 0 0.5rem 0; }
        .project-title { font-size: 1.4rem; font-weight: 600; margin: 0; }
        .project-desc { color: #666; font-size: 0.9rem; margin-top: 0.3rem; }
        .view-block { margin-bottom: 3rem; }
        .view-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.2rem; }
        .view-desc { color: #555; font-size: 0.88rem; margin-bottom: 0.8rem; font-style: italic; }
        .controls-row { margin-bottom: 1rem; }
        .project-select { margin-bottom: 1.5rem; }
    """),
    ui.div(
        ui.h2("My Data Science Projects", class_="project-title"),
        class_="project-header"
    ),
    ui.hr(),
    ui.div(
        ui.input_select(
            id="project_select",
            label="Select Project",
            choices=portfolio.get_titles(),
            selected=portfolio.get_titles()[0]
        ),
        class_="project-select"
    ),
    ui.output_ui("project_content")
)


# ── Server ─────────────────────────────────────────────────

def server(input, output, session):

    @reactive.calc
    def current_project():
        return portfolio.get_project(input.project_select())

    @output
    @render.ui
    def project_content():
        p = current_project()

        if not p.views:
            return ui.div(
                ui.h3(p.title),
                ui.p(p.description),
                ui.p("🚧 Coming soon — check back later!", style="color: #888;")
            )

        blocks = [
            ui.div(
                ui.p(p.description, class_="project-desc"),
                ui.hr()
            )
        ]

        for view in p.views:
            controls = view.get_controls()
            block = ui.div(
                ui.h4(view.title, class_="view-title"),
                ui.p(view.description, class_="view-desc"),
                ui.div(*controls, class_="controls-row") if controls else ui.div(),
                ui.output_ui(f"plot_{view.title.replace(' ', '_')}"),
                class_="view-block"
            )
            blocks.append(block)

        return ui.div(*blocks)

    def _make_renderer(view):
        plot_id = f"plot_{view.title.replace(' ', '_')}"

        @output(id=plot_id)
        @render.ui
        def _plot():
            kwargs = {}
            try:
                kwargs["overview_year"] = input.overview_year()
            except Exception:
                pass
            try:
                kwargs["corr_var"] = input.corr_var()
            except Exception:
                pass
            return ui.HTML(view.render(**kwargs))

    for view in election.views:
        _make_renderer(view)


app = App(app_ui, server)