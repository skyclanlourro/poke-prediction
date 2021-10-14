# -*- coding: utf-8 -*-
# library
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from mymodel import *

# ------------------------------------------------------------------------------
# 

# load data
df = pd.read_csv("./data/pokemon.csv")

# print(df.columns)


df_ds = df.copy()
df_ds["Generation"] = df_ds["Generation"].astype(object)

features = ["Total",
            "Generation",
            "Legendary",
            "Attack",
            "Defense",
            "Sp. Atk",
            "Sp. Def",
            "HP"]

clf, acc = build_model(df, features)

type1 = pd.Series(df["Type 1"]).unique()

# templates used in graphs
graph_template = "plotly_dark"

layout_style = {"background-color": "#800400"}

# ------------------------------------------------------------------------------
# app
app = dash.Dash(
    "DashPokemon",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.SLATE],
)
app.title = "Baby Yoda Pokemon Master"

app_name = "Baby Yoda Pokemon Master"

server = app.server


# --------------------------------------------------------------------------------------------
controls1 = dbc.Form(
    [
        dbc.FormGroup(
            [
                dbc.Label("Type de Pokemon`"),
                dcc.Dropdown(
                    id="dropdown_type",
                    options=[{"label": i, "value": i} for i in type1],
                    value="Fire",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Critère"),
                dcc.Dropdown(
                    id="dropdown_criteria",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "Total",
                            "HP",
                            "Attack",
                            "Defense",
                            "Sp. Atk",
                            "Sp. Def",
                            "Speed",
                        ]
                    ],
                    value="Total",
                ),
            ]
        ),
    ],
    style={"inline-block": True},
)

controls2_a = dbc.Form(
    [
        dbc.FormGroup(
            [
                dbc.Label("Critère de force"),
                dcc.Dropdown(
                    id="strength",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "Total",
                            "HP",
                            "Attack",
                            "Defense",
                            "Sp. Atk",
                            "Sp. Def",
                            "Speed",
                        ]
                    ],
                    value="Total",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Stat type"),
                dcc.RadioItems(
                    id="stat_type",
                    options=[{"label": i, "value": i} for i in ["median", "max"]],
                    value="median",
                    inputStyle={"margin-left": "20px"},
                ),
            ]
        ),
    ]
)

controls2_b = dbc.FormGroup(
    [
        dbc.Label("Couleur"),
        dcc.RadioItems(
            id="color_tab2",
            options=[{"label": i, "value": i} for i in ["Legendary", "Type 1"]],
            value="Type 1",
            inputStyle={"margin-left": "20px"},
        ),
    ]
)


controls3_a = dbc.Form(
    [
        dbc.FormGroup(
            [
                dbc.Label("x axis"),
                dcc.Dropdown(
                    id="x_axis",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "Total",
                            "HP",
                            "Attack",
                            "Defense",
                            "Sp. Atk",
                            "Sp. Def",
                            "Speed",
                        ]
                    ],
                    value="Attack",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("y axis"),
                dcc.Dropdown(
                    id="y_axis",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "Total",
                            "HP",
                            "Attack",
                            "Defense",
                            "Sp. Atk",
                            "Sp. Def",
                            "Speed",
                        ]
                    ],
                    value="Defense",
                ),
            ]
        ),
    ]
)

controls3_b = dbc.FormGroup(
    [
        dbc.Label("Couleur"),
        dcc.RadioItems(
            id="color",
            options=[
                {"label": i, "value": i} for i in ["Legendary", "Type 1", "Generation"]
            ],
            value="Legendary",
            inputStyle={"margin-left": "20px"},
        ),
    ]
)

controls4 = dbc.Form(
    [
        dbc.FormGroup(
            [
                html.H5("Séletionner un Pokemon pour prédire son type"),
                dbc.Label("Nom du pokemon"),
                dcc.Dropdown(
                    id="pokemon_name",
                    options=[[{"label": i, "value": i} for i in ["Gastly", "Haunter"]]],
                    value="Haunter",
                ),
            ]
        ),
        html.Hr(),
        dbc.FormGroup(
            [
                dbc.Label("Types du Pokemon - à inclure dans les données"),
                dcc.Checklist(
                    id="pokemon_type",
                    options=[{"label": i, "value": i} for i in type1],
                    value=["Ground", "Ghost"],
                    inputStyle={"margin-left": "20px"},
                ),
            ]
        ),
        html.Hr(),
        dbc.FormGroup(
            [
                dbc.Label("Features"),
                dcc.Checklist(
                    id = "features_checklist",
                    options=[
                        {"label": "Total", "value": "Total"},
                        {"label": "Generation", "value": "Generation"},
                        {"label": "Legendary", "value": "Legendary"},
                        {"label": "Attack", "value": "Attack"},
                        {"label": "Defense", "value": "Defense"},
                        {"label": "Sp. Atk", "value": "Sp. Atk"},
                        {"label": "Sp. Def", "value": "Sp. Def"},
                        {"label": "HP", "value": "HP"},
                        {"label": "Type 2", "value": "Type 2"},
                    ],
                    value=[
                        "Total",
                        "Generation",
                        "Legendary",
                        "Attack",
                        "Defense",
                        "Sp. Atk",
                        "Sp. Def",
                        "HP",
                    ],
                    inputStyle={"margin-left": "20px"},
                ),
            ]
        ),
    ]
)

# ----------------------------------------------------------------------------
tab1_content = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(controls1, md=2),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H6(id="p_text")),
                                    dbc.CardBody(
                                        html.H4(id="b_type", style={"color": "white"})
                                    ),
                                ],
                                color="info",
                                inverse=True,
                            ),
                            md=3,
                            style={"margin-top": 20},
                        ),
                    ]
                ),
                html.Br(),
                dbc.Card(
                    dbc.Row(
                        [
                            dbc.Col(
                                [dcc.Graph(id="generation", figure="fig"), html.Br()],
                                md=6,
                            ),
                            dbc.Col(dcc.Graph(id="legendary", figure="fig"), md=6),
                        ]
                    )
                ),
            ]
        )
    ]
)


tab2_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(controls2_a, md=2),
                    dbc.Col(
                        [controls2_b, html.Br(), html.H5(id="tab2_criteria"),], md=2
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H6("Le type le plus faible est:")),
                                dbc.CardBody(
                                    html.H4(id="weakest_type", style={"color": "white"})
                                ),
                            ],
                            color="danger",
                            inverse=True,
                        )
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H6("Le type le plsu fort est:")),
                                dbc.CardBody(
                                    html.H4(
                                        id="strongest_type", style={"color": "white"}
                                    )
                                ),
                            ],
                            color="success",
                            inverse=True,
                        )
                    ),
                ]
            ),
            html.Br(),
            dbc.Col(dcc.Graph(id="strength_type", figure="fig"), md=12),
        ]
    )
)

tab3_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            controls3_a,
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H6(id="corr_text")),
                                    dbc.CardBody(
                                        html.H4(
                                            id="correlation", style={"color": "white"}
                                        )
                                    ),
                                ],
                                color="info",
                                inverse=True,
                            ),
                            html.Br(),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(id="scatter_legendary", figure="fig"),
                            html.Br(),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(id="scatter_type", figure="fig"),
                            html.Br(),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(id="scatter_generation", figure="fig"),
                            html.Br(),
                        ],
                        md=3,
                    ),
                ]
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(controls3_b, md=2),
                    dbc.Col(dcc.Graph(id="heatmap", figure="fig"), md=8),
                ]
            ),
        ]
    )
)

tab4_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(controls4, md=3),
                    html.Br(),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.H6("Le type prédit pour"),
                                            html.H6(id="p_name"),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        html.H4(
                                            id="predict_text", style={"color": "white"}
                                        )
                                    ),
                                ],
                                color="info",
                                inverse=True,
                            ),
                            html.Br(),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H6("Model Accuracy:")),
                                    dbc.CardBody(
                                        html.H4(
                                            id="model_acc", style={"color": "white"}
                                        )
                                    ),
                                ],
                                color="info",
                                inverse=True,
                            ),
                            html.Br(),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [dcc.Graph(id="pokemon_stat", figure="fig"), html.Br()], md=3
                    ),
                    dbc.Col(dcc.Graph(id="vc_type", figure="fig"), md=3),
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    html.H4("""Types de Pokemon sélectionné:"""),
                    html.H4(id="ptype_list", style={"color": "white"}),
                ]
            ),
            dbc.Row(
                [
                    html.H4("""Selected Features:"""),
                    html.H4(id="f_checklist", style={"color": "white"}),
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    dcc.Markdown(
                        """
                             * Nous avons utilisé l'algorithme Random Forest Algorythme pour classer les types de Pokémon.
                             * Sélectionnez un pokémon pour voir sa valeur de type prédite.
                             * Ce modèle fonctionne avec une précision de 100% lors de la classification des Pokémons de type Ghost vs Ground (sélectionnez tous les champs sauf Type 2 pour obtenir ce résultat)
                             * La classe dans cet ensemble de données est déséquilibrée.
                             * Par conséquent, sélectionnez seulement 2 types pour obtenir les meilleurs résultats.
                             * La précision du modèle diminuera à mesure que vous sélectionnerez plus de types de Pokémon.
                             * La liste des noms de Pokémon sera mise à jour en fonction des types sélectionnés.
                             * Plus vous sélectionnez de fonctionnalités, meilleures sont les performances du modèle.
                                """
                    ),
                ]
            ),
        ]
    )
)

# ---------------------------------------------------------------------------------
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                html.H3(app_name, style={"display": "inline-block", "color": "white"}),
                html.Img(
                    src=app.get_asset_url("pika.png"),
                    height=50,
                    style={
                        "margin-left": 20,
                        "margin-bottom": 20,
                        "display": "inline-block",
                        "border-radius": 50,
                    },
                ),
                dbc.Col(
                    [
                        dbc.Badge(
                            "instagram",
                            href="https://www.instagram.com/maryrosine/",
                            color="secondary",
                            style={
                                "color": "white",
                                "margin-top": 5,
                                "margin-right": 10,
                                "font-size": 15,
                                "border": "solid 1px white",
                                "float": "right",
                            },
                        ),
                        dbc.Badge(
                            "instagram",
                            href="https://www.instagram.com/skyclanlourro/",
                            color="secondary",
                            style={
                                "color": "white",
                                "margin-top": 5,
                                "margin-right": 10,
                                "font-size": 15,
                                "border": "solid 1px white",
                                "float": "right",
                            },
                        ),
                    ]
                ),
            ]
        ),
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label="Meilleur Pokemon"),
                dbc.Tab(tab2_content, label="Plus fort et plus faible Pokemon"),
                dbc.Tab(tab3_content, label="Attack vs  Defense"),
                dbc.Tab(tab4_content, label="Prédiction du type de Pokemon"),
            ]
        ),
    ],
    fluid=True,
    style=layout_style,
)

# ------------------------------------------------------------------------------------

@app.callback(
    [
        Output("generation", "figure"),
        Output("p_text", "children"),
        Output("b_type", "children"),
    ],
    [Input("dropdown_type", "value"), Input("dropdown_criteria", "value")],
)
def update_fig(dropdown_type, dropdown_criteria):
    filtered_df = df[df["Type 1"] == dropdown_type]
    fig = px.bar(
        filtered_df,
        y="Name",
        x=dropdown_criteria,
        color="Generation",
        template=graph_template,
        height=500,
    )
    best_pokemon = filtered_df.loc[filtered_df[dropdown_criteria].idxmax()].Name
    text = "Pokemon de type {dropdown_type} avec la meilleur {dropdown_criteria} est : "
    fig.update_traces(textfont_size=30)

    fig.update_layout(
        title="Six Generations de pokemons",
        uniformtext_minsize=15,
        transition_duration=500,
    )

    return fig, text, best_pokemon


@app.callback(
    Output("legendary", "figure"),
    [Input("dropdown_type", "value"), Input("dropdown_criteria", "value")],
)
def update_type_compare_fig(dropdown_type, dropdown_criteria):
    filtered_df = df[df["Type 1"] == dropdown_type]

    fig = px.bar(
        filtered_df,
        y="Name",
        x=dropdown_criteria,
        color="Legendary",
        template=graph_template,
        height=500,
    )

    fig.update_traces(textfont_size=30)

    fig.update_layout(
        title="Legendary/Non-legendary pokemons ",
        uniformtext_minsize=15,
        uniformtext_mode="hide",
        transition_duration=500,
    )

    return fig


@app.callback(
    [
        Output("strength_type", "figure"),
        Output("tab2_criteria", "children"),
        Output("weakest_type", "children"),
        Output("strongest_type", "children"),
    ],
    [
        Input("strength", "value"),
        Input("color_tab2", "value"),
        Input("stat_type", "value"),
    ],
)
def strength_fig(strength, color_tab2, stat_type):

    fig = px.box(
        df,
        x=strength,
        y="Type 1",
        points="all",
        hover_name="Name",
        color=color_tab2,
        template=graph_template,
    )


    fig.update_layout(
        title="Boxplot pour different types de pokemon",
        uniformtext_minsize=15,
        uniformtext_mode="hide",
        transition_duration=500,
        height=600,
    )

    if stat_type == "median":
        type_medians = {
            type: df[df["Type 1"] == type][strength].median()
            for type in df["Type 1"].unique()
        }
    elif stat_type == "max":
        type_medians = {
            type: df[df["Type 1"] == type][strength].max()
            for type in df["Type 1"].unique()
        }

    key_max = max(type_medians.keys(), key=(lambda k: type_medians[k]))
    key_min = min(type_medians.keys(), key=(lambda k: type_medians[k]))

    tab2_criteria = "Based on the {stat_type} {strength} value:"

    return fig, tab2_criteria, key_min, key_max


@app.callback(
    [Output("corr_text", "children"), Output("correlation", "children")],
    [Input("x_axis", "value"), Input("y_axis", "value")],
)
def correlation_value(x_axis, y_axis):

    corr_text = "Correlation between {x_axis} and {y_axis} is:"
    correlation = round(df[x_axis].corr(df[y_axis]), 2)
    return corr_text, correlation


@app.callback(
    [
        Output("scatter_legendary", "figure"),
        Output("scatter_type", "figure"),
        Output("scatter_generation", "figure"),
    ],
    [Input("x_axis", "value"), Input("y_axis", "value")],
)
def correlation_scatterplots(x_axis, y_axis):

    # scatterplots
    scatter_legendary = px.scatter(
        df_ds,
        x=x_axis,
        y=y_axis,
        color="Legendary",
        hover_name="Name",
        hover_data=["Attack", "Defense", "HP", "Total"],
        template=graph_template,
        height=300,
        title="Correlation, by legendary",
    )

    scatter_type = px.scatter(
        df_ds,
        x=x_axis,
        y=y_axis,
        color="Type 1",
        hover_name="Name",
        hover_data=["Attack", "Defense", "HP", "Total"],
        template=graph_template,
        height=300,
        title="Correlation,by types",
    )
    scatter_generation = px.scatter(
        df_ds,
        x=x_axis,
        y=y_axis,
        color="Generation",
        hover_name="Name",
        hover_data=["Attack", "Defense", "HP", "Total"],
        template=graph_template,
        height=300,
        title="Correlation,by Genrations",
    )

    return scatter_legendary, scatter_type, scatter_generation


@app.callback(Output("heatmap", "figure"), [Input("color", "value")])
def heatmap_fig(color):

    fig = px.scatter_matrix(
        df,
        dimensions=["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"],
        color=color,
        hover_name="Name",
        template=graph_template,
        height=600,
    )

    fig.update_layout(
        title="Pairplot montrant la relation entre different features de pokemon",
        uniformtext_minsize=15,
        uniformtext_mode="hide",
        transition_duration=500,
    )

    return fig


@app.callback(Output("pokemon_name", "options"), [Input("pokemon_type", "value")])
def update_pokemon_type(pokemon_type):
    df_gd = df[df["Type 1"].isin(pokemon_type)]
    p_names = pd.Series(df_gd["Name"]).unique()
    pokemon_names = [{"label": i, "value": i} for i in p_names]
    return pokemon_names


@app.callback(
    [
        Output("p_name", "children"),
        Output("predict_text", "children"),
        Output("model_acc", "children"),
        Output("f_checklist", "children"),
        Output("ptype_list", "children"),
        Output("pokemon_stat", "figure"),
        Output("vc_type", "figure"),
    ],
    [
        Input("pokemon_name", "value"),
        Input("features_checklist", "value"),
        Input("pokemon_type", "value"),
    ],
)
def prediction_update(pokemon_name, features_checklist, pokemon_type):
    print("Call model")
    df_gd = df[df["Type 1"].isin(pokemon_type)]
    
    clf, acc = build_model(df_gd, features_checklist)
    print("acc =",acc)
    # Model prediction
    p_type, fig = type_prediction(
        pokemon_name, df_gd, clf, features_checklist
    )

    p_name = pokemon_name

    model_acc = round(acc * 100, 2)
    f_checklist = [v + str(", ") for v in features_checklist]
    ptype_list = [v + str(", ") for v in pokemon_type]

    data = df[df["Name"] == pokemon_name].values
    cols = list(df[df["Name"] == pokemon_name].columns)
    fig = px.bar(
        data, x=cols[4:11], y=data[0][4:11], template=graph_template, height=300
    )

    fig.update_traces(textfont_size=30)
    fig.update_layout(
        title="Pokemon stat",
        uniformtext_minsize=15,
        uniformtext_mode="hide",
        transition_duration=50,
    )

    vc = df_gd["Type 1"].value_counts()
    vc_types = list(vc.index)
    vc_counts = list(vc.values)
    vc_type_fig = px.pie(
        vc, values=vc_counts, names=vc_types, template=graph_template, height=300
    )
    vc_type_fig.update_traces(hole=0.4, textfont_size=30)
    vc_type_fig.update_layout(
        title="Selected Pokemon types share",
        uniformtext_minsize=15,
        uniformtext_mode="hide",
        transition_duration=50,
    )

    return p_name, p_type, model_acc, f_checklist, ptype_list, fig, vc_type_fig


# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=False)