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
import shinywatch as watch

class Project:
    '''Represents a single data science project. Each project has metadata and knows how to render its own visualizations'''

    def __init__(self, title: str, description: str, tools: list[str]):
        self.title = title 
        self.description = description
        self.tools = tools
    
    def get_tools_str(self) -> str:
        return " · ".join(self.tools)
    
    def get_data(self) -> pd.DataFrame:
        raise NotImplementedError
    
    def render_plot(self, **kwargs): 
        raise NotImplementedError
    
class PlaceholderProject(Project):
    """A template project with random data. Replace this with your real project by subclassing Project"""

    def __init__(self, title, description, tools):
        super().__init__(title, description, tools)
        np.random.seed(42)
        self._data = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "group": np.random.choice(["A", "B", "C"], 100)
        })

    def get_data(self) -> pd.DataFrame:
        return self._data

    def render_plot(self, color_by: str = "group"):
        df = self.get_data()
        fig = px.scatter(
            df, x="x", y="y", color=color_by,
            title=self.title,
            template="plotly_white"
        )
        return fig

class Portfolio:
    """
    Manages a collection of Project objects. Adding a new project = instantiate a new class and append here.
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
    
# Populate Portfolio 

portfolio = Portfolio()

portfolio.add_project(PlaceholderProject(
    title="Project 1: Exploratory Analysis",
    description="An example project showcasing data exploration and scatter plot visualization.",
    tools=["Python", "Pandas", "Plotly"]
))

portfolio.add_project(PlaceholderProject(
    title="Project 2: Coming Soon",
    description="This slot is reserved for your next data science project.",
    tools=["Python", "Scikit-learn", "Plotly"]
))

# Shiny UI 

app_ui = ui.page_fluid(
    watch.theme.flatly(),
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
            ui.output_text_verbatim("project_info")
        ),
        ui.card(
            ui.output_ui("project_plot")
        )
    )
)

# Shiny Server

def server(input, output, session):

    @reactive.calc
    def selected_project() -> Project:
        return portfolio.get_project(input.project_select())

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
        fig = p.render_plot()
        import plotly.io as pio
        html = pio.to_html(fig, full_html=False)
        return ui.HTML(html)


app = App(app_ui, server)