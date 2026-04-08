"""
Portfolio Shiny App
-------------------
OOP-based structure: each data science project is represented
as a Project object. The Portfolio class manages all projects
and serves them to the Shiny UI.
"""

import pandas as pd
import numpy as np
import plotly.express as px
from shiny import App, ui, render, reactive
import shinyswatch


# ── OOP Layer ──────────────────────────────────────────────

class Project:
    """
    Base class for all portfolio projects.
    Each subclass represents one data science project.
    """

    def __init__(self, title: str, description: str, tools: list[str]):
        self.title = title
        self.description = description
        self.tools = tools

    def get_tools_str(self) -> str:
        return " · ".join(self.tools)

    def get_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_controls(self):
        """Returns Shiny UI controls specific to this project."""
        return []

    def render_plot(self, **kwargs):
        raise NotImplementedError


class ElectionProject(Project):
    """
    Project 1: US County-Level Election Outcomes
    Analyzes how socioeconomic factors influence election results.
    """

    def __init__(self):
        super().__init__(
            title="US County Election Outcomes",
            description=(
                "Analyzes how socioeconomic factors such as GDP, education, "
                "and race influence U.S. election outcomes at the county level (2005-2018)."
            ),
            tools=["Python", "Pandas", "Scikit-learn", "XGBoost", "Plotly"]
        )
        self._data = None

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            url = (
                "https://raw.githubusercontent.com/Jargalsaikhan0528/"
                "us-county-election-outcomes/main/county_census_and_election_result.csv"
            )
            self._data = pd.read_csv(url)
        return self._data

    def get_controls(self):
        df = self.get_data()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        # Remove columns that are not interesting to plot
        exclude = ["year", "FIPS", "fips"]
        options = [c for c in numeric_cols if c not in exclude]
        return [
            ui.input_select(
                id="x_var",
                label="X axis (socioeconomic factor)",
                choices=options,
                selected=options[0] if options else None
            ),
            ui.input_select(
                id="y_var",
                label="Y axis",
                choices=options,
                selected=options[1] if len(options) > 1 else None
            )
        ]

    def render_plot(self, x_var=None, y_var=None):
        df = self.get_data()
        if x_var is None or y_var is None:
            return px.scatter(title="Select variables to explore")
        fig = px.scatter(
            df, x=x_var, y=y_var,
            opacity=0.5,
            title=f"{y_var} vs {x_var} by County",
            template="plotly_white",
            labels={x_var: x_var, y_var: y_var}
        )
        return fig


class PlaceholderProject(Project):
    """
    Template for future projects.
    Replace with a real subclass when ready.
    """

    def __init__(self, title, description, tools):
        super().__init__(title, description, tools)

    def get_data(self) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "group": np.random.choice(["A", "B", "C"], 100)
        })

    def render_plot(self, **kwargs):
        df = self.get_data()
        return px.scatter(
            df, x="x", y="y", color="group",
            title=self.title,
            template="plotly_white"
        )


class Portfolio:
    """
    Manages all Project objects.
    To add a new project: instantiate its class and call add_project().
    """

    def __init__(self):
        self._projects: list[Project] = []

    def add_project(self, project: Project):
        self._projects.append(project)

    def get_titles(self) -> list[str]:
        return [p.title for p in self._projects]

    def get_project(self, title: str) -> Project:
        for p in self._projects:
            if p.title == title:
                return p
        raise ValueError(f"Project '{title}' not found.")


# ── Populate Portfolio ──────────────────────────────────────

portfolio = Portfolio()

portfolio.add_project(ElectionProject())

portfolio.add_project(PlaceholderProject(
    title="Project 2: Coming Soon",
    description="This slot is reserved for your next data science project.",
    tools=["Python", "Pandas", "Plotly"]
))

portfolio.add_project(PlaceholderProject(
    title="Project 3: Coming Soon",
    description="This slot is reserved for your third data science project.",
    tools=["Python", "Pandas", "Plotly"]
))


# ── Shiny UI ────────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.h2("📊 My Data Science Projects"),
    ui.hr(),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Select a Project"),
            ui.input_select(
                id="project_select",
                label="Project",
                choices=portfolio.get_titles()
            ),
            ui.hr(),
            ui.output_ui("dynamic_controls"),
            ui.hr(),
            ui.output_text_verbatim("project_info")
        ),
        ui.card(
            ui.output_ui("project_plot")
        )
    ),
    theme=shinyswatch.theme.flatly()
)


# ── Shiny Server ────────────────────────────────────────────

def server(input, output, session):

    @reactive.calc
    def selected_project() -> Project:
        return portfolio.get_project(input.project_select())

    @output
    @render.ui
    def dynamic_controls():
        p = selected_project()
        controls = p.get_controls()
        return ui.TagList(*controls)

    @output
    @render.text
    def project_info():
        p = selected_project()
        return (
            f"Description:\n{p.description}\n\n"
            f"Tools: {p.get_tools_str()}"
        )

    @output
    @render.ui
    def project_plot():
        p = selected_project()
        kwargs = {}
        if hasattr(input, "x_var"):
            try:
                kwargs["x_var"] = input.x_var()
                kwargs["y_var"] = input.y_var()
            except Exception:
                pass
        fig = p.render_plot(**kwargs)
        import plotly.io as pio
        html = pio.to_html(fig, full_html=False)
        return ui.HTML(html)


app = App(app_ui, server)